"""Contradiction hints only: never replace a local opinion with a context opinion."""
from food_category_review import CATEGORIES
CONTEXT_PROMPT='Inspect all clearly visible food in the entire image. Ignore subtitles, people and utensils. Return exactly one word: meat, vegetable, mixed, other, or unknown. Meat is animal flesh including fish; vegetable includes plant ingredients, fruit and garlic. Use mixed if both animal flesh and plant ingredients are visibly present anywhere, even in separate containers. Use other for different recognizable food, unknown if no clear food or evidence is insufficient. Do not infer raw/cooked status, physical contact, cutting-board purpose or hygiene compliance.'
def compare_categories(local,context):
    if local not in CATEGORIES or context not in CATEGORIES:raise ValueError('Expected canonical food categories')
    sets={'meat':{'meat'},'vegetable':{'vegetable'},'mixed':{'meat','vegetable'}}
    if 'unknown' in [local,context]:relation='insufficient_evidence'
    elif 'other' in [local,context]:relation='not_comparable'
    else:relation='not_contradicted' if sets[local].issubset(sets[context]) else 'contradiction'
    return {'local_category':local,'context_category':context,'relation':relation,'category_replaced':False,
            'status':'needs_review','compliance_determined':False}
def review_food_context(image,regions,predict):
    rows=[dict(r) for r in regions]
    relevant=[r for r in rows if r.get('category_inference_performed')]
    review={'status':'needs_review','context_category':'unknown','context_inference_performed':False,
            'scope':'Separate full-image opinion from the SAME model, not independent-model evidence or contact verification.'}
    if not any(r.get('food_category') not in [None,'unknown'] for r in relevant):
        review['reason']='No non-unknown local category for comparison';return rows,review
    try:
        review['context_inference_performed']=True;raw=predict(image);review['context_raw_response']=raw;value=raw.strip().lower()
        if value in CATEGORIES:review['context_category']=value
        else:review['context_error']='invalid_context_response'
    except Exception as exc:review['context_error']=repr(exc)
    for r in relevant:r['category_context_comparison']=compare_categories(r['food_category'],review['context_category'])
    review['contradiction_count']=sum(r.get('category_context_comparison',{}).get('relation')=='contradiction' for r in rows)
    review['compared_regions']=len(relevant)
    return rows,review
