"""Build model-authored silver traces for the twelve pinned MSC train conversations.

Human review is required. Unlisted turns become explicit, high-priority
no-update review items; this default is never treated as gold supervision.
No raw MSC utterances are stored in the output.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
V1 = HERE / 'msc_graph_traces_v1.json'
OUT = HERE / 'msc_graph_traces_v2.json'

# Audit order is SHA256-sorted by the pinned MSC queue builder. Session lengths
# are checked against the downloaded source by the replay test.
CONVERSATIONS = {
    1: ('train:ordered_2281', (12, 12, 11)),
    2: ('train:ordered_4419', (12, 12)),
    3: ('train:ordered_6210', (12, 12)),
    4: ('train:ordered_3280', (12, 12)),
    5: ('train:ordered_7807', (12, 12)),
    6: ('train:ordered_5704', (12, 10)),
    8: ('train:ordered_6469', (12, 12)),
    9: ('train:ordered_6340', (12, 12)),
    10: ('train:ordered_6729', (12, 12)),
    11: ('train:ordered_792', (12, 12, 12)),
}
# Assignment is at conversation level. Held-out partitions stay out of writer
# fitting, including the earlier 100 attachment pairs (0, 1, 2, 7, 10, 11).
DEV = {'train:ordered_7807', 'train:ordered_5704'}
HOLDOUT = {'train:ordered_6469', 'train:ordered_6340'}
ANNOTATIONS = {}


def _add(i, ref, subject, action, **fields):
    rows = ANNOTATIONS.setdefault(i, {})
    if ref in rows:
        raise ValueError((i, ref))
    rows[ref] = dict(ref=ref, subject=subject, action=action,
                     evidence=fields.pop('evidence', fields.get('state', '')), **fields)


def root(i, ref, subject, key, state):
    _add(i, ref, subject, 'create_root', identity=key, scope=key,
         state=state, segment=True)


def child(i, ref, subject, key, parent, state):
    _add(i, ref, subject, 'create_child', identity=key, scope=key,
         parent_key=parent, state=state, segment=True)


def attach(i, ref, subject, key, state, segment=False):
    _add(i, ref, subject, 'attach', target_keys=[key], state=state, segment=segment)


def multi(i, ref, subject, updates):
    _add(i, ref, subject, 'multi_attach', target_keys=list(updates),
         updates=[dict(target_key=k, state=v) for k, v in updates.items()],
         evidence='One turn updates each named process')


# 1: separate dog ownership, dog/cat compatibility, book, and rabbit management.
root(1,'s2:t0',0,'Dog ownership','Takes a dog for walks; dog chased a squirrel')
attach(1,'s2:t2',0,'Dog ownership','Dog chased a squirrel for around ten minutes without catching it')
root(1,'s2:t5',1,'Bella the cat','Owns a cat who usually watches through the window')
root(1,'s2:t7',1,'Considering a dog','Considering another pet dog and whether it would get along with Bella')
root(1,'s2:t10',0,'Cat allergy','Highly allergic to cats and cannot keep one in the house')
root(1,'s2:t11',1,'Bible reading','Has not recently read the Bible and intends to resume reading')
root(1,'s3:t1',1,'Writing a book','Writing a new book')
attach(1,'s3:t2',0,'Dog ownership','Owns Waylon, a black Labrador who has killed wild rabbits')
attach(1,'s3:t3',1,'Considering a dog','May call a shelter to find a dog suited to cats')
attach(1,'s3:t4',0,'Dog ownership','Got Waylon a few months earlier; he is settling in and friendly to people')
attach(1,'s3:t5',1,'Bella the cat','Bella is a rescue cat owned for seven years')
child(1,'s3:t6',0,'Protecting yard rabbits','Dog ownership','Takes Waylon out on a leash so wild rabbits can escape the yard')
attach(1,'s3:t8',0,'Protecting yard rabbits','Leash strategy seems to help; rabbits may be learning to avoid yard')
attach(1,'s4:t0',0,'Dog ownership','Took Waylon for a hike')
attach(1,'s4:t2',0,'Dog ownership','Waylon enjoys running and hiking with owner')
attach(1,'s4:t4',0,'Dog ownership','Waylon is five years old and enjoys hiking')

# 2: family, outdoor activity, degree and medical school are distinct processes.
root(2,'s2:t0',0,'Fishing with children','Took children fishing together')
attach(2,'s2:t2',0,'Fishing with children','Older children help younger ones bait hooks; family caught fish for a picnic')
root(2,'s2:t3',1,'Recent relocation','Moved to a warmer area closer to ponds and lakes')
root(2,'s2:t4',0,'Raising children','Children spend time outdoors in a warm climate')
child(2,'s2:t5',1,'Family fishing plans','Recent relocation','Plans to buy fishing gear in spring after moving near ponds')
attach(2,'s2:t6',0,'Raising children','Children swim often during hot summers and spend much time outside')
root(2,'s2:t7',1,'Pre-med degree','In a difficult new pre-med semester and looking forward to clinical work')
child(2,'s2:t9',1,'Medical school planning','Pre-med degree','Near end of undergraduate study; expects further medical education')
root(2,'s2:t10',0,'Music education','Majored in music at college and did not pursue postgraduate study')
attach(2,'s2:t11',1,'Medical school planning','Undecided whether to stay locally for medical school or seek a stronger school elsewhere')
attach(2,'s3:t2',0,'Raising children','Makes individual quality time for each child')
root(2,'s3:t3',1,'Growing family','Has two children and would like more')
attach(2,'s3:t4',0,'Raising children','Has at least three children and finds a large family busy')
attach(2,'s3:t5',1,'Family fishing plans','Hopes to take children fishing and teach them skills')
attach(2,'s3:t6',0,'Fishing with children','Four children enjoy fishing; another skips the trips and may feel excluded')
attach(2,'s3:t7',1,'Growing family','Juggles work, school and quality time with two children')
child(2,'s3:t8',0,'Children’s music lessons','Raising children','Trying group music lessons so all children can take part together')
attach(2,'s3:t10',0,'Children’s music lessons','Children are trying piano; all interested and two showing aptitude')

# 3: avoid treating another person’s shopping or a one-off dinner as deep goals.
root(3,'s2:t0',0,'Gifts for friend’s family','Bought shirts for friend’s children')
attach(3,'s2:t2',0,'Gifts for friend’s family','Agrees to look for a shirt for friend’s wife')
attach(3,'s2:t4',0,'Gifts for friend’s family','Plans to find a shirt and pink lip gloss for friend’s wife')
root(3,'s2:t6',0,'Fashion shopping','Recently purchased several Gucci pieces')
root(3,'s2:t7',1,'Baking for family','Bought heart-shaped muffin pans to make muffins for family')
attach(3,'s2:t9',1,'Baking for family','Plans to try banana-pecan muffins and serve friend at dinner')
root(3,'s2:t10',0,'Social connection','Feels lonely while single and values time with friend’s family')
attach(3,'s3:t0',0,'Gifts for friend’s family','Bought lip gloss for friend’s wife')
attach(3,'s3:t2',0,'Fashion shopping','Bought a bag, shoes and shirts')
root(3,'s3:t3',1,'Household subscriptions','Wife uses makeup and workout-clothing subscription boxes')
root(3,'s3:t4',0,'Exploring subscriptions','Considering subscription boxes but undecided among options')
attach(3,'s3:t6',0,'Fashion shopping','Shops both online and in stores, preferring online variety')
attach(3,'s3:t8',0,'Exploring subscriptions','Interested in trying GreenChef using a coupon')
attach(3,'s3:t9',1,'Household subscriptions','Family uses GreenChef; twins prefer plain food; interested in baking box')

# 4: relationship transition and soccer involvement, with a child for a new courtship.
root(4,'s2:t0',0,'Dating and relationships','Ended relationship with boyfriend after meeting Thor')
root(4,'s2:t3',1,'Soccer career','Selected for first team this weekend')
attach(4,'s2:t5',1,'Soccer career','Will play West Ham United in an important match')
root(4,'s2:t6',0,'Attending soccer match','Wants to attend friend’s match subject to work schedule')
attach(4,'s2:t8',0,'Attending soccer match','Plans to ask boss for time off and bring an office friend')
attach(4,'s2:t10',0,'Attending soccer match','Hopes to meet group for lunch before match')
root(4,'s3:t0',0,'Playing soccer','Scored two goals at practice')
attach(4,'s3:t2',0,'Attending soccer match','Still plans to attend match with an office acquaintance')
child(4,'s3:t4',0,'Interest in coworker','Dating and relationships','Trying to impress an office acquaintance at the match')
attach(4,'s3:t6',0,'Interest in coworker','Feels ready to date again and considers coworker sweet')
attach(4,'s3:t8',0,'Interest in coworker','Shares college and musical interests with coworker, who supports them at work')
attach(4,'s3:t10',0,'Interest in coworker','Hopes coworker likes them; cohabitation is only a distant possibility')

# 5: app, arbitrage income, job loss and a separate financial strain process.
root(5,'s2:t0',0,'Lostfound app','Finished developing a new phone app')
attach(5,'s2:t2',0,'Lostfound app','App called Lostfound tracks misplaced objects and sends reminders')
root(5,'s2:t3',1,'Additional income','Considering selling unneeded belongings for extra money')
root(5,'s2:t5',1,'Hiking','Plans a hike the following day')
root(5,'s2:t8',0,'Online resale business','Buys discounted goods and resells on eBay and Amazon for profit')
attach(5,'s2:t9',1,'Additional income','May learn about buying and reselling from friend')
attach(5,'s3:t1',1,'Hiking','Enjoyed hike and took pictures; wants to return')
attach(5,'s3:t2',0,'Online resale business','eBay and Amazon accounts suspended; resale income interrupted',True)
child(5,'s3:t4',0,'App development job','Lostfound app','Reports being laid off from app-related work; relation to ownership of app uncertain')
root(5,'s3:t6',0,'Family financial strain','Worried about ability to support children after losing income')

# 6: move as root, storing belongings and rental as separate narrower processes.
root(6,'s2:t1',1,'Sister’s books','Sister Susan King has published books; newest is speaker’s favorite')
attach(6,'s2:t3',1,'Sister’s books','Sister releases books digitally and avoids paper copies')
root(6,'s2:t6',0,'Move to Australia','Plans to move to Australia; will store possessions in the US')
child(6,'s2:t8',0,'Houston house rental','Move to Australia','Will rent Houston house to a friend while away')
attach(6,'s2:t10',0,'Move to Australia','Expects friends to visit after moving to Australia')
root(6,'s3:t0',0,'Reading Susan King’s book','Started reading friend’s sister’s digital book')
attach(6,'s3:t2',0,'Reading Susan King’s book','Halfway through the book and enjoying it')
attach(6,'s3:t4',0,'Reading Susan King’s book','Intends to read another recommended book when time permits')

# 8: acting, boat business, and shared visit plans have separate subjects.
root(8,'s2:t1',1,'Acting career','Previously acted in The Hobbit; says filming opportunities are scarce')
attach(8,'s2:t3',1,'Acting career','Local work now mostly independent or horror films')
root(8,'s2:t4',0,'Boats and fishing','Interested primarily in boats and fishing rather than film')
attach(8,'s2:t5',1,'Acting career','Short stature limits acting roles, according to speaker')
child(8,'s2:t6',0,'Boat sales','Boats and fishing','Usually sells about five or six boats a year')
root(8,'s2:t7',1,'Visiting friend’s island','Would like to visit friend’s island but gets seasick on boats')
attach(8,'s2:t9',1,'Visiting friend’s island','May try anti-nausea medication; wants to fish on visit')
attach(8,'s2:t11',1,'Visiting friend’s island','May visit for a weekend in a few months and hike')
attach(8,'s3:t0',0,'Boats and fishing','Planning fishing and hiking with visiting friend')
attach(8,'s3:t1',1,'Visiting friend’s island','Wants hiking and shore fishing without boating')
attach(8,'s3:t3',1,'Visiting friend’s island','Prefers a fishing spot on hiking trail and avoids boats')
attach(8,'s3:t4',0,'Boats and fishing','Identified river hiking and fishing spot; offers equipment')
attach(8,'s3:t5',1,'Visiting friend’s island','Needs to borrow fishing equipment and is eager to hike')
root(8,'s3:t7',1,'Recreational fishing','Fishes for leisure but does not eat fish')
root(8,'s3:t8',0,'Dog care','Dog enjoys fish treats')
attach(8,'s3:t10',0,'Dog care','Cleans and cooks fish to avoid feeding dog bones')

# 9: explicit parent/child for family plans and return to work.
root(9,'s2:t1',1,'Raising five children','Has five active children and takes them to park in family car')
root(9,'s2:t2',0,'Raising baby','Has a baby and feels tired caring for the child')
root(9,'s2:t3',1,'Return to employment','Misses working while caring for five children')
root(9,'s2:t4',0,'Flexible employment','Employer offers flexibility after baby was born')
attach(9,'s2:t5',1,'Return to employment','Interested in a flexible position at friend’s company')
root(9,'s2:t6',0,'Concert outing','Invites friend to Bruno Mars concert and offers to arrange sitter')
attach(9,'s2:t7',1,'Raising five children','Finds a large family rewarding but demanding')
child(9,'s2:t8',0,'Planning another child','Raising baby','Discussed another child with spouse; postponing until baby is potty trained')
root(9,'s2:t10',0,'Dogs around baby','Dogs are gentle with baby; wonders how they will respond to toddler behavior')
child(9,'s3:t0',0,'Potty training daughter','Raising baby','Started potty training baby daughter')
attach(9,'s3:t2',0,'Potty training daughter','Potty training is harder than expected')
attach(9,'s3:t4',0,'Potty training daughter','Considering pausing training until daughter is older')
root(9,'s3:t5',1,'Unreliable car','Current car is often unreliable')
root(9,'s3:t6',0,'Family Subaru','Bought a Subaru roomy enough for family and dogs')
child(9,'s3:t7',1,'Remote accounting search','Return to employment','Considering night-time remote accounting to support family and afford car')
attach(9,'s3:t9',1,'Remote accounting search','Has fifteen years of prior accounting experience and asks about online roles')
attach(9,'s3:t11',1,'Remote accounting search','Welcomes friend’s offer to check remote accounting openings')

# 10: relationships, housing, move and travel plans are linked conservatively.
root(10,'s2:t0',0,'Relationship with boyfriend','Boyfriend in Italy booked a visit')
root(10,'s2:t1',1,'Italy travel interest','Wants to visit opera houses and Stradivarius workshop in Italy')
attach(10,'s2:t2',0,'Relationship with boyfriend','Boyfriend lives in Italy; has visited him several times')
attach(10,'s2:t4',0,'Relationship with boyfriend','Boyfriend is an Italian-born chef working at family restaurant')
attach(10,'s2:t6',0,'Relationship with boyfriend','Boyfriend is from Sicily and has relatives throughout Italy')
root(10,'s2:t7',1,'Life with grandmother','Grandmother is energetic; went bowling together')
root(10,'s2:t8',0,'Current housing','Lives with mother and younger brother; hopes to move in with boyfriend')
root(10,'s2:t11',1,'Violin teaching plans','Plays violin daily and hopes to start teaching lessons')
child(10,'s3:t0',0,'Moving in with boyfriend','Relationship with boyfriend','Boyfriend invited speaker to move in')
attach(10,'s3:t1',1,'Life with grandmother','Still lives with grandmother who is from Italy')
attach(10,'s3:t2',0,'Moving in with boyfriend','Plans to start in Sicily, then possibly move to Venice in a few years')
root(10,'s3:t6',0,'Exploring Italy','Wants to see Rome after moving to Italy')
attach(10,'s3:t8',0,'Exploring Italy','Plans to explore Italy gradually, including Colosseum and Trevi Fountain')

# 11: cake business, dance lessons, walks, and dance group evolve separately.
root(11,'s2:t0',0,'Making cakes','Finishing a cake for friend')
root(11,'s2:t3',1,'Dance group','Plans to celebrate group’s strong performance by sharing cake')
root(11,'s2:t4',0,'Learning dance','Considering beginner dance group despite little experience')
root(11,'s2:t6',0,'Nature walks','Enjoys walking in nature')
root(11,'s2:t7',1,'Reading habit','Reads a little but is easily distracted')
root(11,'s2:t11',1,'Church attendance','Attends church each Sunday with parents')
attach(11,'s3:t0',0,'Making cakes','Completed and handed over friend’s cake')
child(11,'s3:t2',0,'Wedding cake business','Making cakes','Perfecting cake skills for wedding season')
attach(11,'s3:t4',0,'Wedding cake business','Spring and summer are busiest for wedding cake work')
attach(11,'s3:t5',1,'Dance group','Offers dance lessons in exchange for cakes')
attach(11,'s3:t6',0,'Learning dance','Interested in trading cakes for dance lessons')
root(11,'s3:t7',1,'Competitive eating','Began competitive eating after a dance show')
root(11,'s3:t10',0,'Harry Potter interest','Follows Harry Potter actors in other shows')
attach(11,'s4:t0',0,'Learning dance','Considering taking friend up on cake-for-dance-lessons offer',True)
attach(11,'s4:t1',1,'Competitive eating','Enjoys cakes and participates in eating competitions')
attach(11,'s4:t2',0,'Wedding cake business','Paid for wedding cakes except when baking for friends or family')
attach(11,'s4:t3',1,'Dance group','Dancing helps offset competitive eating; available for lessons')
attach(11,'s4:t4',0,'Learning dance','Wants lessons soon to prepare for a wedding next Friday')
attach(11,'s4:t6',0,'Nature walks','Using nature walks to get back in shape')
attach(11,'s4:t7',1,'Competitive eating','Wants to attempt Nathan’s hot dog challenge; qualification is difficult')
attach(11,'s4:t11',1,'Competitive eating','Plans to use cake as eating practice for hot dog challenge')


def build():
    v1 = json.loads(V1.read_text())
    traces = {t['conversation_id']: t for t in v1['traces']}
    for i, (conversation_id, session_lengths) in CONVERSATIONS.items():
        planned = ANNOTATIONS[i]
        refs = [f's{session}:t{turn}' for session, count in enumerate(session_lengths, 2)
                for turn in range(count)]
        assert set(planned) <= set(refs), (conversation_id, set(planned)-set(refs))
        keys = {}
        rows = []
        for ref in refs:
            if ref not in planned:
                slot = int(ref.split(':t')[1]) % 2
                rows.append(dict(ref=ref, subject=slot, action='no_update',
                                 evidence='No durable process update selected; inspect original turn',
                                 review_priority='high', label_origin='default_no_update'))
                continue
            row = planned[ref].copy()
            action = row['action']
            if action in ('create_root', 'create_child'):
                key = row['identity']
                assert key not in keys, (conversation_id, ref, key)
                if action == 'create_child':
                    row['parent'] = keys[row.pop('parent_key')]
                keys[key] = len(keys)
            else:
                names = row.pop('target_keys', [])
                if names:
                    row['targets'] = [keys[k] for k in names]
                for u in row.get('updates', []):
                    u['target'] = keys[u.pop('target_key')]
            row['label_origin'] = 'model_authored'
            row['review_priority'] = 'high' if action in ('create_root','create_child','multi_attach') else 'normal'
            rows.append(row)
        traces[conversation_id] = dict(conversation_id=conversation_id, rows=rows)
    for trace in v1['traces']:
        for row in trace['rows']:
            row['label_origin'] = 'model_authored'
            row['review_priority'] = 'high' if row['action'] in ('create_root','multi_attach') else 'normal'
    for conversation_id, trace in traces.items():
        trace['split'] = ('holdout_review' if conversation_id in HOLDOUT else
                          'development' if conversation_id in DEV else 'training_review')
    return dict(source=v1['source'], annotation_status='single-model provisional silver; all defaults and hierarchy require human review',
                format=v1['format'], traces=list(traces.values()))


if __name__ == '__main__':
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + '\n')
