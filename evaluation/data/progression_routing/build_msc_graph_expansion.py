"""Model-authored full MSC expansion traces; source dialogue stays in CI artifact.

These two conversations were selected for genuine updates to two established
processes. Labels are provisional and are not benchmark ground truth.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / 'msc_graph_expansion_v1.json'
CASES = {
    'train:ordered_1628': ('development', (12, 12, 12)),
    'train:ordered_909': ('holdout_review', (12, 12)),
    'train:ordered_7748': ('holdout_review', (12, 12)),
}
UPDATES = {key: {} for key in CASES}


def add(c, ref, subject, action, **fields):
    if ref in UPDATES[c]:
        raise ValueError((c, ref))
    UPDATES[c][ref] = dict(ref=ref, subject=subject, action=action,
                           evidence=fields.pop('evidence', fields.get('state', '')),
                           **fields)


def root(c, ref, subject, key, state):
    add(c, ref, subject, 'create_root', identity=key, scope=key, state=state, segment=True)


def attach(c, ref, subject, key, state, segment=False):
    add(c, ref, subject, 'attach', target_keys=[key], state=state, segment=segment)


def multi(c, ref, subject, updates, evidence):
    add(c, ref, subject, 'multi_attach', target_keys=list(updates),
        updates=[dict(target_key=k, state=v) for k, v in updates.items()], evidence=evidence)


def create_and_attach(c, ref, subject, key, state, updates, evidence):
    add(c, ref, subject, 'create_and_attach', identity=key, scope=key,
        state=state, target_keys=list(updates),
        updates=[dict(target_key=k, state=v) for k, v in updates.items()],
        evidence=evidence, segment=True)


A = 'train:ordered_1628'
root(A,'s2:t0',0,'Art practice','Painted a tree landscape and wants to share it')
root(A,'s2:t2',0,'Marriage strain','Hides time spent painting from wife')
attach(A,'s2:t4',0,'Marriage strain','Wife dismisses speaker’s passion for art as a dead end')
multi(A,'s2:t6',0,{
    'Art practice':'Using saved lottery winnings to pursue art despite uncertainty',
    'Marriage strain':'Feels unsupported by wife while pursuing art'},
    'Speaker links funding and pursuit of art with lack of spousal support')
attach(A,'s2:t8',0,'Marriage strain','Speculates wife may object to time spent on art or hold gendered views')
attach(A,'s2:t10',0,'Marriage strain','Has tried including wife in art to share that part of life')
root(A,'s3:t1',1,'IT work','Helps people set up home offices for remote work and answers their support questions')
attach(A,'s3:t2',0,'Art practice','Working on a new piece in a basement studio and seeks feedback')
attach(A,'s3:t3',1,'IT work','Often handles frustrated support callers by phone or email')
attach(A,'s3:t4',0,'Marriage strain','Believes marriage may not work out; wife views art as idleness')
attach(A,'s3:t6',0,'Art practice','Hopes artwork will gain attention and support future change')
root(A,'s3:t7',1,'Art practice','Creates art inspired by childhood in London and scenery in Austria')
root(A,'s3:t8',0,'Austria visit','Interested in visiting art-making friend in Austria')
attach(A,'s3:t10',0,'Austria visit','Agrees to visit Austria and asks to stay with friend')
multi(A,'s4:t0',0,{
    'Austria visit':'Booked ticket to Austria to visit artist friend',
    'Marriage strain':'Concealed Austria trip from wife by claiming it was a nearby art exhibition'},
    'Booking and cover story update existing travel and marriage processes',)
attach(A,'s4:t2',0,'Marriage strain','Feels wife is uninterested in art and must pursue it alone')
attach(A,'s4:t3',1,'Art practice','Likes exhibitions and artist interaction but hesitates to display own art')
attach(A,'s4:t5',1,'Art practice','Prefers sculpture to painting and doubts quality despite boyfriend’s praise')
attach(A,'s4:t7',1,'Art practice','Started with computer 3D art because IT work allowed practice without a workshop')
attach(A,'s4:t8',0,'Marriage strain','Wife becomes bored discussing art; speaker perceives different interests')
attach(A,'s4:t10',0,'Marriage strain','Uncertain about long-term marriage plans despite shared history')

B = 'train:ordered_909'
root(B,'s2:t0',0,'Divorce','Getting a divorce')
attach(B,'s2:t2',0,'Divorce','Husband says he is no longer in love; speaker fears weight gain contributed')
attach(B,'s2:t4',0,'Divorce','Still has some affection for husband but feels rejected')
attach(B,'s2:t6',0,'Divorce','Trying to remain optimistic through divorce')
root(B,'s2:t7',1,'Mack the dog','Dog Mack damaged garden flowers')
root(B,'s2:t8',0,'Considering a dog','Considering getting a dog after divorce')
root(B,'s2:t10',0,'Post-divorce travel','Interested in visiting Madrid or Tuscany after divorce')
root(B,'s2:t11',1,'Loan repayment','Cannot afford travel this year because loan remains unpaid')
root(B,'s3:t0',0,'Fitness plan','Planned weight loss to feel better independently of divorce',)
multi(B,'s3:t2',0,{
    'Considering a dog':'Considering a dog for companionship and daily exercise',
    'Fitness plan':'Considering dog walks as motivation to exercise during weight loss'},
    'The proposed dog supports companionship and the existing exercise plan')
attach(B,'s3:t3',1,'Mack the dog','Mack is aging but remains affectionate')
attach(B,'s3:t6',0,'Post-divorce travel','Already knows some Spanish and previously traveled in Mexico')
attach(B,'s3:t10',0,'Post-divorce travel','Suggests planning a future Mexico trip together')

C = 'train:ordered_7748'
root(C,'s2:t2',0,'Playing football','Plays quarterback')
root(C,'s2:t3',1,'Life on boat','Lives on a boat in the Atlantic Ocean')
root(C,'s2:t4',0,'Marine biology studies','Studying marine biology in college')
root(C,'s2:t6',0,'Surfing','Surfs in Hawaii and enjoys it')
attach(C,'s2:t8',0,'Surfing','Practices surfing and finds it worthwhile despite difficulty')
create_and_attach(C,'s3:t0',0,'Leg injury','Leg injured during football practice',
                  {'Playing football':'Leg injured at football practice; playing is interrupted'},
                  'One injury begins recovery and interrupts existing football activity')
multi(C,'s3:t4',0,{
    'Playing football':'Cannot currently play football because of injured leg',
    'Surfing':'Cannot currently surf because of injured leg',
    'Marine biology studies':'Plans to focus more on studies while unable to play sports'},
    'Speaker says injured leg prevents both sports and shifts attention to study')
attach(C,'s3:t8',0,'Leg injury','Expects a couple of weeks without walking on injured leg')


def build():
    traces = []
    for conversation_id, (split, lengths) in CASES.items():
        annotated = UPDATES[conversation_id]
        refs = [f's{session}:t{turn}' for session, length in enumerate(lengths, 2)
                for turn in range(length)]
        assert set(annotated) <= set(refs)
        keys = {}
        rows = []
        for ref in refs:
            if ref not in annotated:
                rows.append(dict(ref=ref, subject=int(ref.split(':t')[1])%2,
                                 action='no_update', evidence='No durable process update selected; inspect source',
                                 label_origin='default_no_update', review_priority='high'))
                continue
            row = annotated[ref].copy()
            if row['action'] == 'create_root':
                key = (row['subject'], row['identity'])
                assert key not in keys
                keys[key] = len(keys)
            else:
                row['targets'] = [keys[row['subject'], key] for key in row.pop('target_keys')]
                for update in row.get('updates', []):
                    update['target'] = keys[row['subject'], update.pop('target_key')]
                if row['action'] == 'create_and_attach':
                    key = (row['subject'], row['identity'])
                    assert key not in keys
                    keys[key] = len(keys)
            row.update(label_origin='model_authored',
                       review_priority='high' if row['action'] in ('create_root','multi_attach','create_and_attach') else 'normal')
            if conversation_id == 'train:ordered_909' and ref == 's2:t8':
                row['review_issue'] = ('Turn starts a dog decision and mentions post-divorce travel; '
                                       'current contract cannot create two new progressions from one turn')
                row['review_priority'] = 'high'
            if conversation_id == 'train:ordered_909' and ref == 's2:t10':
                row['review_issue'] = ('Madrid/Tuscany is the first labeled travel progression, '
                                       'but Barcelona was mentioned two turns earlier')
            rows.append(row)
        traces.append(dict(conversation_id=conversation_id, split=split, rows=rows))
    return dict(source='Meta/ParlAI Multi-Session Chat v0.1 train; pinned 48-conversation expansion audit',
                annotation_status='single-model provisional silver; independent review pending',
                traces=traces)


if __name__ == '__main__':
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False)+'\n')
