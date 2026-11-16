"""Page-first transitive group assignment; no writer identity is inferred."""
import hashlib
import json
import random


def components(rows):
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate page ID')
    parent={id_:id_ for id_ in ids}
    def root(x):
        while x!=parent[x]: parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for row in rows:
        for group in row.get('groups',[]):
            if group in seen: parent[root(row['id'])]=root(seen[group])
            seen[group]=row['id']
    groups={}
    for id_ in sorted(ids):groups.setdefault(root(id_),[]).append(id_)
    return sorted(groups.values())


def validate_splits(rows, assignments):
    if set(assignments)!={r['id'] for r in rows}: raise ValueError('split page coverage mismatch')
    if not set(assignments.values()) <= {'train','validation','test'}: raise ValueError('invalid split')
    for group in components(rows):
        if len({assignments[x] for x in group})!=1: raise ValueError(f'group leakage: {group}')


def assign_splits(rows, seed=42):
    groups=components(rows);random.Random(seed).shuffle(groups)
    sizes=dict(train=0,validation=0,test=0);targets=dict(train=.8,validation=.1,test=.1); assignments={}
    # Whole connected components, biggest first. Seed breaks equal-size ties.
    for group in sorted(groups,key=len,reverse=True):
        split=min(sizes,key=lambda k:sizes[k]/targets[k])
        for id_ in group:assignments[id_]=split
        sizes[split]+=len(group)
    validate_splits(rows,assignments)
    return dict(seed=seed,status='provisional_page_group_pilot',writer_independence_verified=False,
                algorithm='transitive group union; seeded largest-first weighted allocation 80/10/10',
                assignments=assignments,components=groups,counts=sizes,
                input_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
