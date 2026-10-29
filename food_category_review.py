"""Optional visual category opinions for board-candidate regions, never hygiene labels."""
import math
FOOD_CATEGORY_PROMPT='Inspect only the visible food in this cropped image. Ignore subtitles and hands or utensils. Return exactly one word: meat, vegetable, mixed, other, or unknown. Meat means animal flesh including fish; vegetable means plant ingredients including fruit and garlic. Use mixed for visibly combined animal flesh and plant ingredients, other for a different recognizable food, and unknown if there is no clear food or visual evidence is insufficient. Do not infer raw/cooked state, food safety, cutting-board purpose or compliance.'
CATEGORIES={'meat','vegetable','mixed','other','unknown'}
def review_board_foods(image,regions,predict,*,compare_tight_crop=False):
    result=[]
    for region in regions:
        r=dict(region);r.update(food_category='unknown',category_inference_performed=False,category_review_status='not_board_candidate')
        if r['location']!='board_candidate':result.append(r);continue
        x1,y1,x2,y2=r['box'];w=x2-x1;h=y2-y1
        if min(w,h)<20:r['category_review_status']='crop_too_small';result.append(r);continue
        crop=[max(0,math.floor(x1-.1*w)),max(0,math.floor(y1-.1*h)),min(image.width,math.ceil(x2+.1*w)),min(image.height,math.ceil(y2+.1*h))]
        r['category_crop_xyxy']=crop
        r['category_evidence_scope']='visible_food_in_expanded_crop'
        r['category_margin_fraction']=.1
        r['center_object_identity_verified']=False
        r['category_does_not_verify_contact']=True
        try:
            r['category_inference_performed']=True;raw=predict(image.crop(crop));r['category_raw_response']=raw
            value=raw.strip().lower()
            if value in CATEGORIES:r.update(food_category=value,category_review_status='needs_review')
            else:r['category_review_status']='invalid_category_response'
        except Exception as exc:r.update(category_review_status='category_inference_failed',category_error=repr(exc))
        if compare_tight_crop:
            tight=[max(0,math.floor(x1)),max(0,math.floor(y1)),min(image.width,math.ceil(x2)),min(image.height,math.ceil(y2))]
            comparison={'tight_crop_xyxy':tight,'tight_opinion':'unknown','status':'needs_review','original_opinion_preserved':True,'detection_removed':False}
            try:
                raw=predict(image.crop(tight));comparison['raw_response']=raw
                value=raw.strip().lower()
                if value in CATEGORIES:comparison['tight_opinion']=value
                else:comparison['status']='invalid_category_response'
                a=r['food_category'];b=comparison['tight_opinion']
                comparison['relation']='insufficient_evidence' if 'unknown' in (a,b) else ('same_category' if a==b else 'different_category')
            except Exception as exc:comparison.update(status='category_inference_failed',error=repr(exc),relation='insufficient_evidence')
            r['tight_crop_comparison']=comparison
        result.append(r)
    return result
