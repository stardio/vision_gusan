import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox
import sys
from datetime import datetime
from collections import deque

class VisionInspector:
    def __init__(self, master_path):
        self.path = r'C:\VisionMaster'
        self.log_path = os.path.join(self.path, 'NG_Log')
        self.master_path = master_path
        self.config_file = os.path.join(self.path, 'config.json')
        
        for p in [self.path, self.log_path]:
            if not os.path.exists(p): os.makedirs(p)

        self.img_m = cv2.imread(master_path)
        if self.img_m is None: self.img_m = np.zeros((1080, 1920, 3), dtype=np.uint8)

        self.roi_list = []
        self.th_shape = 80.0
        self.th_surface = 85.0
        self.gain_val = 0
        self.draw_mode = 0 
        self.load_config()
        self.cap = self._init_camera()
        
        self.is_dragging = False
        self.start_point = (0, 0); self.curr_point = (0, 0)
        self.inspection_mode = False
        self.roi_offsets = {}
        self.track_margin = 30
        self.track_min_score = 0.45
        self.last_ng_time = 0 
        
        self.CLR = {'bg': (30, 30, 30), 'btn_default': (70, 70, 70), 'btn_active': (0, 150, 0),
                    'btn_reset': (60, 60, 180), 'ok': (0, 220, 0), 'ng': (0, 0, 255),
                    'guide': (255, 255, 0), 'master': (200, 100, 0), 'thresh': (100, 100, 200)}

    def _init_camera(self):
        for i in [0, 1, 2]:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_GAIN, self.gain_val); return cap
        return None

    def get_shape_score(self, img_m_roi, img_t_roi, r_type):
        try:
            def get_cnt(img, rt):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (9, 9), 0)
                _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                h, w = th.shape
                mask = np.zeros((h, w), dtype=np.uint8)
                if rt == 0: mask.fill(255)
                else: cv2.circle(mask, (w//2, h//2), min(w, h)//2, 255, -1)
                th = cv2.bitwise_and(th, mask)
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return max(cnts, key=cv2.contourArea) if cnts else None
            cnt_m = get_cnt(img_m_roi, r_type)
            cnt_t = get_cnt(img_t_roi, r_type)
            if cnt_m is None or cnt_t is None: return 0
            ret = cv2.matchShapes(cnt_m, cnt_t, cv2.CONTOURS_MATCH_I1, 0.0)
            return max(0, 100 - (ret * 500))
        except: return 0

    def get_surface_score(self, img_m_roi, img_t_roi, r_type):
        try:
            gray_m, gray_t = cv2.cvtColor(img_m_roi, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_t_roi, cv2.COLOR_BGR2GRAY)
            diff_m = np.mean(gray_m) - np.mean(gray_t)
            gray_t_fixed = cv2.convertScaleAbs(gray_t, alpha=1, beta=diff_m)
            diff = cv2.absdiff(cv2.GaussianBlur(gray_m,(5,5),0), cv2.GaussianBlur(gray_t_fixed,(5,5),0))
            _, diff_th = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY) 
            h, w = diff_th.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            if r_type == 0: mask.fill(255)
            else: cv2.circle(mask, (w//2, h//2), min(w, h)//2, 255, -1)
            diff_final = cv2.bitwise_and(diff_th, mask)
            total_area = cv2.countNonZero(mask)
            score = max(0, 100 - ((cv2.countNonZero(diff_final) / total_area) * 500)) 
            return score
        except: return 0

    def save_ng_image(self, frame):
        now = datetime.now()
        if (now.timestamp() - self.last_ng_time) > 3:
            filename = now.strftime("NG_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(self.log_path, filename), frame)
            self.last_ng_time = now.timestamp()

    def _clamp_roi(self, x, y, w, h, img_w, img_h):
        if w <= 0 or h <= 0 or w > img_w or h > img_h:
            return None
        x = max(0, min(x, img_w - w))
        y = max(0, min(y, img_h - h))
        return x, y

    def _reset_tracking(self):
        self.roi_offsets = {i: (0, 0) for i in range(len(self.roi_list))}

    def _track_roi(self, idx, canvas_cam, rx, ry, rw, rh):
        img_h, img_w = canvas_cam.shape[:2]
        base_dx, base_dy = self.roi_offsets.get(idx, (0, 0))
        base_x, base_y = rx + base_dx, ry + base_dy
        clamped = self._clamp_roi(base_x, base_y, rw, rh, img_w, img_h)
        if clamped is None:
            return rx, ry
        base_x, base_y = clamped

        m_roi = self.img_m[ry:ry+rh, rx:rx+rw]
        if m_roi.size == 0:
            return base_x, base_y

        sx1 = max(0, base_x - self.track_margin)
        sy1 = max(0, base_y - self.track_margin)
        sx2 = min(img_w, base_x + rw + self.track_margin)
        sy2 = min(img_h, base_y + rh + self.track_margin)
        search = canvas_cam[sy1:sy2, sx1:sx2]

        if search.shape[0] < rh or search.shape[1] < rw:
            return base_x, base_y

        template_gray = cv2.cvtColor(m_roi, cv2.COLOR_BGR2GRAY)
        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.GaussianBlur(template_gray, (5, 5), 0)
        search_gray = cv2.GaussianBlur(search_gray, (5, 5), 0)

        res = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.track_min_score:
            return base_x, base_y

        tracked_x = sx1 + max_loc[0]
        tracked_y = sy1 + max_loc[1]

        max_shift = self.track_margin * 3
        new_dx = int(np.clip(tracked_x - rx, -max_shift, max_shift))
        new_dy = int(np.clip(tracked_y - ry, -max_shift, max_shift))

        smooth_dx = int(0.7 * base_dx + 0.3 * new_dx)
        smooth_dy = int(0.7 * base_dy + 0.3 * new_dy)
        self.roi_offsets[idx] = (smooth_dx, smooth_dy)

        final = self._clamp_roi(rx + smooth_dx, ry + smooth_dy, rw, rh, img_w, img_h)
        if final is None:
            return base_x, base_y
        return final

    def run(self):
        win_name = "VISION MASTER PRO v4.5"
        cv2.namedWindow(win_name); cv2.setMouseCallback(win_name, self.on_mouse)
        cv2.createTrackbar("GAIN", win_name, self.gain_val, 100, lambda v: (setattr(self, 'gain_val', v), self.cap.set(cv2.CAP_PROP_GAIN, v)))

        while True:
            ret, frame_raw = self.cap.read()
            if not ret: break
            h_m, w_m = self.img_m.shape[:2]
            aspect = frame_raw.shape[1] / frame_raw.shape[0]
            nw, nh = w_m, int(w_m / aspect)
            resized_cam = cv2.resize(frame_raw, (nw, nh))
            canvas_cam = np.zeros((h_m, w_m, 3), dtype=np.uint8)
            yo, xo = (h_m - nh) // 2, (w_m - nw) // 2
            canvas_cam[yo:yo+nh, xo:xo+nw] = resized_cam; self.last_canvas = canvas_cam.copy()

            main_view = np.hstack((self.img_m.copy(), canvas_cam))
            status_bar = np.zeros((40, main_view.shape[1], 3), dtype=np.uint8)
            cv2.putText(status_bar, f"S-TH: {self.th_shape}% | F-TH: {self.th_surface}% | LOG: {self.log_path}", (20, 27), 0, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            main_view = np.vstack((status_bar, main_view))

            # 1. 드래그 중일 때 참조라인 실시간 표시
            if self.is_dragging:
                sx, sy = self.start_point; cx, cy = self.curr_point
                if self.draw_mode == 0:
                    cv2.rectangle(main_view, (sx, sy+40), (cx, cy+40), self.CLR['guide'], 1)
                else:
                    r = int(np.sqrt((sx-cx)**2 + (sy-cy)**2))
                    cv2.circle(main_view, (sx, sy+40), r, self.CLR['guide'], 1)

            has_ng = False
            for i, (rx, ry, rw, rh, rt) in enumerate(self.roi_list):
                oy = 40
                if rt == 0: cv2.rectangle(main_view, (rx, ry+oy), (rx+rw, ry+rh+oy), self.CLR['guide'], 1)
                else: cv2.circle(main_view, (rx+rw//2, ry+rh//2+oy), rw//2, self.CLR['guide'], 1)
                
                if self.inspection_mode:
                    tx, ty = self._track_roi(i, canvas_cam, rx, ry, rw, rh)
                    m_roi = self.img_m[ry:ry+rh, rx:rx+rw]
                    t_roi = canvas_cam[ty:ty+rh, tx:tx+rw]
                    if m_roi.size == 0 or t_roi.size == 0 or m_roi.shape[:2] != t_roi.shape[:2]:
                        continue
                    
                    s_shp, s_srf = self.get_shape_score(m_roi, t_roi, rt), self.get_surface_score(m_roi, t_roi, rt)
                    is_ok = s_shp >= self.th_shape and s_srf >= self.th_surface
                    if not is_ok: has_ng = True

                    ex, ey = tx + w_m, ty + oy
                    color = self.CLR['ok'] if is_ok else self.CLR['ng']
                    if rt == 0: cv2.rectangle(main_view, (ex, ey), (ex+rw, ey+rh), color, 2)
                    else: cv2.circle(main_view, (ex+rw//2, ey+rh//2), rw//2, color, 2)
                    
                    cv2.putText(main_view, f"V1:{s_shp:.0f}%", (ex, ey-22), 0, 0.35, color, 1)
                    cv2.putText(main_view, f"V2:{s_srf:.0f}%", (ex, ey-8), 0, 0.35, color, 1)

            if self.inspection_mode and has_ng: self.save_ng_image(canvas_cam)

            btn_bar = np.zeros((80, main_view.shape[1], 3), dtype=np.uint8)
            btn_bar[:] = self.CLR['bg']
            self._draw_button(btn_bar, 10, 80, "RESET", self.CLR['btn_reset'])
            self._draw_button(btn_bar, 90, 160, "RECT", (180, 90, 0), active=(self.draw_mode==0))
            self._draw_button(btn_bar, 170, 240, "CIRCLE", (0, 130, 180), active=(self.draw_mode==1))
            self._draw_button(btn_bar, 250, 360, "SET MASTER", self.CLR['master'])
            self._draw_button(btn_bar, 370, 480, "DUAL THRESH", self.CLR['thresh'])
            self._draw_button(btn_bar, 490, 590, "START", self.CLR['btn_active'] if self.inspection_mode else (70, 70, 70))
            self._draw_button(btn_bar, main_view.shape[1]-100, main_view.shape[1]-10, "EXIT", (50, 50, 50))
            
            cv2.imshow(win_name, np.vstack((main_view, btn_bar)))
            if cv2.waitKey(1) == 27: break
        self.cap.release(); cv2.destroyAllWindows()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.th_shape = data.get("th_shape", 80.0); self.th_surface = data.get("th_surface", 85.0)
                self.roi_list = [tuple(r) for r in data.get("roi_list", [])]; self.gain_val = data.get("gain", 0)

    def save_config(self):
        data = {"th_shape": self.th_shape, "th_surface": self.th_surface, "roi_list": self.roi_list, "gain": self.gain_val}
        with open(self.config_file, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)

    def _draw_button(self, bar, x1, x2, text, color, active=False):
        cv2.rectangle(bar, (x1, 10), (x2, 60), color, -1, cv2.LINE_AA)
        if active: cv2.rectangle(bar, (x1, 10), (x2, 60), (255, 255, 255), 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(text, 0, 0.35, 1)
        cv2.putText(bar, text, (x1 + (x2-x1-tw)//2, 40), 0, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    def on_mouse(self, event, x, y, flags, param):
        ry = y - 40
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > self.img_m.shape[0] + 40: self._handle_btns(x)
            elif y > 40 and x < self.img_m.shape[1]:
                self.is_dragging, self.start_point = True, (x, ry); self.curr_point = (x, ry)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging: self.curr_point = (x, ry)
        elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
            self.is_dragging = False; sx, sy = self.start_point
            if self.draw_mode == 0:
                rw, rh = abs(x-sx), abs(ry-sy)
                if rw > 10:
                    self.roi_list.append((min(x, sx), min(ry, sy), rw, rh, 0))
                    self._reset_tracking()
                    self.save_config()
            else:
                radius = int(np.sqrt((sx-x)**2 + (sy-ry)**2))
                if radius > 5:
                    self.roi_list.append((sx-radius, sy-radius, radius*2, radius*2, 1))
                    self._reset_tracking()
                    self.save_config()

    def _handle_btns(self, x):
        if 10 <= x <= 80:
            self.roi_list = []
            self.inspection_mode = False
            self._reset_tracking()
            self.save_config()
        elif 90 <= x <= 160: self.draw_mode = 0
        elif 170 <= x <= 240: self.draw_mode = 1
        elif 250 <= x <= 360: 
            self.img_m = self.last_canvas.copy(); self._reset_tracking(); cv2.imwrite(self.master_path, self.img_m); self.save_config()
        elif 370 <= x <= 480: 
            root = tk.Tk(); root.withdraw()
            s = simpledialog.askfloat("S-TH", "찌그러짐(%)", initialvalue=self.th_shape)
            if s: f = simpledialog.askfloat("F-TH", "찍힘(%)", initialvalue=self.th_surface)
            if s and f: self.th_shape, self.th_surface = s, f; self.save_config()
            root.destroy()
        elif 490 <= x <= 590:
            self.inspection_mode = not self.inspection_mode if self.roi_list else False
            if self.inspection_mode:
                self._reset_tracking()
        elif x > (self.img_m.shape[1] * 2 - 100): sys.exit()

if __name__ == "__main__":
    inspector = VisionInspector(r'C:\VisionMaster\master.png')
    inspector.run()