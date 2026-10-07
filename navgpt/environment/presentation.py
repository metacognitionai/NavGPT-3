"""Matched canvases and coordinate transforms for the spatial-interface study."""
import math
import numpy as np
from PIL import Image

COMPACT = {
 '0':(7,5,5,5,7),'1':(2,6,2,2,7),'2':(7,1,7,4,7),'3':(7,1,7,1,7),
 '4':(5,5,7,1,1),'5':(7,4,7,1,7),'6':(7,4,7,5,7),'7':(7,1,1,1,1),
 '8':(7,5,7,5,7),'9':(7,5,7,1,7),
}


def compact_digit(img, ch, x, y, width, height, color):
    """Resize 3x5 digit to the reference glyph bounding box (same contrast/height)."""
    mask = np.array([[(bits >> (2-c)) & 1 for c in range(3)] for bits in COMPACT[ch]],dtype=np.uint8)
    mask = np.asarray(Image.fromarray(mask).resize((width,height),Image.NEAREST)).astype(bool)
    x0,y0,x1,y1=max(0,x),max(0,y),min(img.shape[1],x+width),min(img.shape[0],y+height)
    if x1>x0 and y1>y0:
        region=img[y0:y1,x0:x1];region[mask[y0-y:y1-y,x0-x:x1-x]]=color


def camera_canvas(raw, labels, ticks, variant, text_bars, labels_at):
    # Same reserved padding in every condition; no scene pixels overwritten.
    h,w=raw.shape[:2];top,bottom=128,64
    canvas=np.zeros((h+top+bottom,w,3),dtype=np.uint8)
    canvas[:]=(10,12,16);canvas[top:top+h]=raw
    if variant!='text_labels':
        text_bars(canvas[:top], labels, scales=[4,2])
        labels_at(canvas[top+h:],[(u*w,t) for u,t in ticks])
    return canvas, {'labels':labels,'bearing_ticks':[{'u':u,'text':t} for u,t in ticks],
                    'scene_box':[0,top,w,h], 'presentation':variant}


def rotate_map(array, points, heading_rad=0.0):
    """Pad all variants to identical diagonal squares; rotate pixels and handles together."""
    h,w=array.shape[:2];side=int(math.ceil(math.hypot(w,h)))+4
    dx,dy=(side-w)//2,(side-h)//2
    canvas=Image.new('RGB',(side,side),(10,12,16));canvas.paste(Image.fromarray(array),(dx,dy))
    # Habitat yaw positive is left; image rotation negative yaw puts heading up.
    angle=-math.degrees(heading_rad)
    canvas=canvas.rotate(angle,resample=Image.NEAREST,expand=False)
    a=math.radians(angle);c,s=math.cos(a),math.sin(a);center=side/2
    def point(p):
        x,y=p[0]+dx-center,p[1]+dy-center
        return [center+c*x+s*y,center-s*x+c*y]
    return np.asarray(canvas),{k:point(v) for k,v in points.items()}, {'rotation_deg':angle,'padding_xy':[dx,dy],'source_shape':[h,w]}
