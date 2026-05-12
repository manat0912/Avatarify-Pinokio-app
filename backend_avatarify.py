import sys, os
import io
import time
import argparse
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'fomm'))

from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn
import cv2
import numpy as np
import yaml
import glob
from afy.utils import resize, crop

app = FastAPI()

class AvatarifyState:
    def __init__(self):
        self.predictor = None
        self.avatars = []
        self.avatar_names = []
        self.cur_ava = 0
        self.avatar = None
        self.avatar_kp = None
        self.IMG_SIZE = 512
        self.frame_proportion = 0.95
        self.frame_offset_x = 0
        self.frame_offset_y = 0
        self.is_calibrated = False
        self.auto_track = False
        self.tracker_center = None
        
    def load_images(self, img_size=512):
        self.avatars = []
        self.avatar_names = []
        images_list = sorted(glob.glob('./avatars/*'))
        for f in images_list:
            if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'):
                img = cv2.imread(f)
                if img is None:
                    continue
                if img.ndim == 2:
                    img = np.tile(img[..., None], [1, 1, 3])
                img = img[..., :3][..., ::-1]
                img = resize(img, (img_size, img_size))
                self.avatars.append(img)
                self.avatar_names.append(os.path.basename(f))
        if self.avatars:
            self.change_avatar(0)
            
    def change_avatar(self, idx):
        if idx < 0 or idx >= len(self.avatars): return
        self.cur_ava = idx
        self.avatar = self.avatars[idx]
        if self.predictor:
            self.avatar_kp = self.predictor.get_frame_kp(self.avatar)
            self.predictor.set_source_image(self.avatar)
            
    def init_predictor(self, mode='fomm', enhance=False):
        predictor_args = {
            'relative': True,
            'adapt_movement_scale': True,
            'smooth_factor': 0.9,
            'jacobian_dampening': 0.3,
            'jacobian_stabilization': 1.0,
            'no_relative_jacobian': False,
            'enhance': enhance
        }
        if mode == 'fomm':
            predictor_args['config_path'] = 'fomm/config/vox-adv-256.yaml'
            predictor_args['checkpoint_path'] = 'vox-adv-cpk.pth.tar'
            predictor_args['enc_downscale'] = 1.0
            from afy import predictor_local
            self.predictor = predictor_local.PredictorLocal(**predictor_args)
        elif mode == 'face-animation':
            predictor_args['config_path'] = None
            predictor_args['checkpoint_path'] = None
            predictor_args['enc_downscale'] = 1.0
            from afy import predictor_face_animation
            self.predictor = predictor_face_animation.PredictorFaceAnimation(**predictor_args)
            
        self.load_images()
        
    def process_frame(self, frame_orig):
        if not self.predictor:
            return frame_orig

        if self.auto_track:
            full_lmk = self.predictor.get_frame_landmarks(frame_orig)
            if full_lmk is not None:
                face_center = full_lmk.mean(axis=0)
                target_cx = int(face_center[0])
                target_cy = int(face_center[1])
                if self.tracker_center is None:
                    self.tracker_center = [target_cx, target_cy]
                else:
                    self.tracker_center[0] = 0.2 * self.tracker_center[0] + 0.8 * target_cx
                    self.tracker_center[1] = 0.2 * self.tracker_center[1] + 0.8 * target_cy
            
            if self.tracker_center is not None:
                frame, (ox, oy) = crop(frame_orig, p=self.frame_proportion, center=self.tracker_center, offset_x=self.frame_offset_x, offset_y=self.frame_offset_y)
            else:
                frame, (ox, oy) = crop(frame_orig, p=self.frame_proportion, offset_x=self.frame_offset_x, offset_y=self.frame_offset_y)
        else:
            frame, (ox, oy) = crop(frame_orig, p=self.frame_proportion, offset_x=self.frame_offset_x, offset_y=self.frame_offset_y)

        frame = resize(frame, (self.IMG_SIZE, self.IMG_SIZE))[..., :3]

        if not self.is_calibrated:
            # We must calibrate first. We can auto-calibrate on first frame.
            self.calibrate(frame_orig)

        out = self.predictor.predict(frame)
        if out is None:
            return frame
        return out

    def calibrate(self, frame_orig):
        self.predictor.reset_frames()
        full_lmk = self.predictor.get_frame_landmarks(frame_orig)
        if full_lmk is not None:
            face_center = full_lmk.mean(axis=0)
            self.tracker_center = [int(face_center[0]), int(face_center[1])]
        self.is_calibrated = True

state = AvatarifyState()

@app.post("/init")
async def init_model(request: Request):
    data = await request.json()
    mode = data.get('mode', 'fomm')
    enhance = data.get('enhance', False)
    state.init_predictor(mode=mode, enhance=enhance)
    return {"status": "ok", "avatars": state.avatar_names}

@app.post("/control")
async def control(request: Request):
    data = await request.json()
    action = data.get('action')
    if action == 'calibrate':
        state.is_calibrated = False
    elif action == 'auto_track':
        state.auto_track = data.get('value', False)
        if not state.auto_track:
            state.tracker_center = None
    elif action == 'zoom_in':
        state.frame_proportion = max(state.frame_proportion - 0.05, 0.1)
    elif action == 'zoom_out':
        state.frame_proportion = min(state.frame_proportion + 0.05, 1.0)
    elif action == 'change_avatar':
        idx = data.get('index', 0)
        state.change_avatar(idx)
    return {"status": "ok"}

@app.post("/process")
async def process(request: Request):
    # Read raw jpeg bytes
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response(status_code=400)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out_frame = state.process_frame(frame)
    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
    
    _, encoded_img = cv2.imencode('.jpg', out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
