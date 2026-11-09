"""Long-horizon decision continuity pilot: fixed V2 engine, no LLM calls.

Six fictional career trajectories inspired by common long-running career,
graduate-study, research, entrepreneurship, and financial trade-offs. The
test uses explicit checkpoint gold values *only for scoring*, never as
retrieval context. These are deterministic state-access tests, not evidence
of planner quality or live model writer accuracy.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from progression_memory.benchmarks.v2_paired import TraceFixture
from progression_memory.benchmarks.v2_retrieval_ablation import prepare
from progression_memory.evaluation.retrieval import BM25Index, RetrievalDocument
from progression_memory.v2.models import AttributeOperation, AttributePatch, Observation
from progression_memory.v2.read_path import build_documents
from progression_memory.v2.writer_pipeline import ProposedUpdate, WriterProposal

PERSONAS = (
    ("founder", "startup building", "salary and financial runway", "research mentorship"),
    ("systems", "distributed systems expertise", "remote location flexibility", "technical ownership"),
    ("quant", "quantitative research", "geographic proximity", "market learning"),
    ("research", "publishable AI research", "financial stability", "strong scientific mentorship"),
    ("product", "shipping useful products", "predictable employment", "startup autonomy"),
    ("education", "graduate study for applied ML", "tuition affordability", "hands-on engineering"),
)
STAGES = (
    ("exploration", "exploring options", "technical depth", "limited project experience"),
    ("internship", "completed an internship", "technical ownership", "time available for research"),
    ("research", "started applied research", "research mentorship", "uncertain publication output"),
    ("offer", "received a return offer", "financial stability", "decision deadline approaching"),
    ("reassessment", "reconsidered the prior study plan", "financial runway", "tuition cost is high"),
    ("commitment", "accepted a job offer", "honor signed commitments", "switching requires notice"),
    ("reversal", "explicitly deprioritized a former path", "practical building", "avoid immediate full-time study"),
    ("project", "started a technical venture", "startup learning", "project needs sustained effort"),
    ("constraint", "a family constraint changed availability", "location flexibility", "travel must be limited"),
    ("opportunity", "received a new competing opportunity", "career optionality", "cannot take both roles"),
    ("planning", "made a six-month plan", "measurable progress", "avoid overscheduling"),
    ("review", "reviewed progress and blockers", "technical depth", "maintain financial runway"),
)


def make_fixture(*, distractors: int = 8):
    if distractors < 0:
        raise ValueError("distractors must be nonnegative")
    start = datetime(2023, 1, 1, tzinfo=UTC)
    observations = []
    proposals = {}
    checkpoints = []
    for slug, ambition, initial_constraint, alternative in PERSONAS:
        subject = f"fictional:{slug}"
        pid = f"v2:{slug}-root:0"
        def add(oid, day, text, proposal, *, subject=subject):
            observation = Observation(id=oid, source_id=oid, subject_id=subject,
                                      text=text, recorded_at=start+timedelta(days=day))
            observations.append(observation)
            proposals[oid] = WriterProposal(updates=(proposal,) if proposal else ())
        add(f"{slug}-root", 0,
            f"My enduring goal is {ambition}. I need to balance it against {initial_constraint}.",
            ProposedUpdate(action="create", name=f"{slug} career development",
                reason="explicit long-term personal goal",
                patches=(AttributePatch(key="long_term_goal",
                    operation=AttributeOperation.SET, value=ambition),)))
        for idx, (phase, event, priority, constraint) in enumerate(STAGES):
            day = 25+idx*65
            if idx in (4,6):
                # The benchmark explicitly tests recently superseded priorities.
                event += f". I explicitly changed priorities and no longer prioritize {alternative} now"
            oid=f"{slug}-update-{idx}"
            add(oid, day,
                f"Career checkpoint: {event}. Current top priority: {priority}. "
                f"Active constraint: {constraint}. Long-term goal remains {ambition}.",
                ProposedUpdate(action="transition", progression_id=pid,
                    reason="explicit updated goal or constraint", stage=phase,
                    patches=(AttributePatch(key="priority",operation=AttributeOperation.SET,value=priority),
                             AttributePatch(key="constraint",operation=AttributeOperation.SET,value=constraint),
                             AttributePatch(key="phase",operation=AttributeOperation.SET,value=phase))))
            for j in range(distractors):
                add(f"{slug}-chat-{idx}-{j}", day+j+1,
                    f"Someone mentioned {alternative} and {initial_constraint} in a "
                    f"hypothetical discussion about possible career paths. "
                    f"No personal decision or changed priority was made.",None)
            if idx in (3,4,5,6,8,9,11):
                checkpoints.append({"id":f"decision-{slug}-{idx}",
                    "subject_id":subject, "as_of": (start+timedelta(days=day+1)).isoformat(),
                    "prompt":"What goal, priority and constraint should guide the next career decision?",
                    "gold":{"long_term_goal":ambition,"priority":priority,"constraint":constraint},
                    "evidence_id":oid})
    observations.sort(key=lambda o:(o.recorded_at,o.id))
    fixture=TraceFixture.model_validate({"observations":[o.model_dump(mode="json") for o in observations],
                "proposals":{k:v.model_dump(mode="json") for k,v in proposals.items()},
                "questions":[]})
    return fixture,checkpoints


def score(*,distractors=8,budgets=(400,800,1600,3200)):
    fixture,checkpoints=make_fixture(distractors=distractors)
    engine=prepare(fixture)
    systems=("rag_bm25","v2_state_bm25","v2_full_bm25","v2_oracle_state")
    output={}
    for system in systems:
        output[system]={}
        for budget in budgets:
            count=0
            per=[]
            for q in checkpoints:
                expected=q["gold"]
                subject=q["subject_id"]
                horizon=datetime.fromisoformat(q["as_of"])
                if system=="v2_oracle_state":
                    # Upper bound on structured memory accessing the correct
                    # process as-of time, without retrieval or an LLM.
                    pid=f"v2:{subject.split(':')[-1]}-root:0"
                    actual=engine.state_at(pid,valid_at=horizon,known_at=horizon)
                    present={k:actual.get(k)==v for k,v in expected.items()}
                else:
                    if system=="rag_bm25":
                        documents=[
                            RetrievalDocument(id=o.id,text=o.text,
                                              evidence_ids=frozenset({o.id}))
                            for o in fixture.observations
                            if o.subject_id==subject and o.recorded_at<=horizon]
                    else:
                        documents=list(build_documents(engine,subject_id=subject,
                            view="state" if system=="v2_state_bm25" else "full",
                            valid_at=horizon,known_at=horizon))
                    ranked=BM25Index(documents).retrieve(q["prompt"],len(documents))
                    context=[]
                    chars=0
                    for doc in ranked:
                        if chars+len(doc.text)>budget:
                            break
                        context.append(doc.text)
                        chars+=len(doc.text)
                    text="\n".join(context)
                    present={k:(v in text) for k,v in expected.items()}
                hits=sum(present.values())
                count+=hits
                per.append({"question_id":q["id"],"fields_present":hits,
                            "all_fields_present":hits==len(expected)})
            output[system][str(budget)]={"field_visibility_accuracy":count/(3*len(checkpoints)),
                "complete_decision_context_rate":sum(p["all_fields_present"] for p in per)/len(per),
                "per_checkpoint":per}
    return {"dataset":{"personas":len(PERSONAS),"observations":len(fixture.observations),
                       "checkpoints":len(checkpoints),"horizon_days":25+(len(STAGES)-1)*65,
                       "distractors_per_stage":distractors},
            "systems":output,
            "limitations":[
                "Deterministic answer-field visibility, not LLM-generated decision quality.",
                "All updates are scripted oracle writer proposals.",
                "V2 oracle state uses exact progression ID; upper bound, not live retrieval.",
                "Scenarios reuse a common phase schema, not independent natural conversations.",
                "Gold labels are withheld from retrieval but values recur in input observations."
            ]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--distractors",type=int,default=8)
    args=p.parse_args()
    result=score(distractors=args.distractors)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"dataset":result["dataset"],"at_1600":{
        key:round(value["1600"]["field_visibility_accuracy"],4)
        for key,value in result["systems"].items()}},indent=2))


if __name__=="__main__":
    main()
