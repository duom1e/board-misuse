"""Optional identity opinions; never veto geometry or downstream detections."""
import math
from review_detector_candidates import PROMPT
LABELS={'cutting_board','food_container','table_surface','other','unknown'}
def review_boards(image,boxes,predict):
    rows=[]
    for i,box in enumerate(boxes):
        x,y,z,k=box;w=z-x;h=k-y
        crop=[max(0,math.floor(x-.15*w)),max(0,math.floor(y-.15*h)),min(image.width,math.ceil(z+.15*w)),min(image.height,math.ceil(k+.15*h))]
        row=dict(board_index=i,box=list(box),crop_box=crop,opinion='unknown',status='needs_review',detection_removed=False,compliance_determined=False)
        try:
            raw=predict(image.crop(crop));value=raw.strip().lower();row.update(raw_response=raw,opinion=value if value in LABELS else 'unknown',response_valid=value in LABELS)
        except Exception as exc:row['error']=repr(exc)
        rows.append(row)
    return rows
