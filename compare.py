import cv2
import numpy as np
import os
import json
import csv
import tkinter as tk
from tkinter import simpledialog, messagebox
import sys
import time
from datetime import datetime
from collections import deque

class VisionInspector:
    def __init__(self, master_path):
        self.path = r'C:\VisionMaster'
        self.log_path = os.path.join(self.path, 'NG_Log')
        self.csv_path = os.path.join(self.path, 'inspect_log.csv')
        self.master_path = master_path
        self.config_file = os.path.join(self.path, 'config.json')
        
        for p in [self.path, self.log_path]:
            if not os.path.exists(p): os.makedirs(p)

        self.img_m = cv2.imread(master_path)
        if self.img_m is None: self.img_m = np.zeros((1080, 1920, 3), dtype=np.uint8)

        self.roi_list = []
        self.th_match = 90.0
        self.gain_val = 0
        self.cam_params = {
            "gain": 0.0,
            "exposure": -6.0,
            "brightness": 128.0,
            "contrast": 32.0,
            "saturation": 64.0,
            "wb_temperature": 4500.0,
            "auto_exposure": False,
            "auto_wb": True,
        }
        self.draw_mode = 0 
        self.load_config()
        self.cam_index = self._select_camera_index()
        self.cap = self._init_camera(self.cam_index)
        self.apply_camera_params(save=False)
        self.cam_param_ui = None
        
        self.is_dragging = False
        self.start_point = (0, 0); self.curr_point = (0, 0)
        self.inspection_mode = False
        self.inspect_triggered = False
        self.last_result = None
        self.last_result_time = 0.0
        self.result_hold_sec = 2.0
        self.display_scale = (1.0, 1.0)
        self.roi_offsets = {}
        self.track_margin = 30
        self.track_min_score = 0.45
        self.last_ng_time = 0 
        self.frame_buffer = deque(maxlen=5)
        
        self.CLR = {'bg': (30, 30, 30), 'btn_default': (70, 70, 70), 'btn_active': (0, 150, 0),
                'btn_reset': (60, 60, 180), 'ok': (0, 220, 0), 'ng': (0, 0, 255),
                'guide': (255, 255, 0), 'master': (200, 100, 0), 'thresh': (100, 100, 200),
                'warn': (0, 200, 200)}

    def _select_camera_index(self):
        available = []
        for i in [0, 1, 2, 3]:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        if not available:
            return 0
        if len(available) == 1:
            return available[0]
        root = tk.Tk(); root.withdraw()
        msg = "사용할 카메라 번호를 선택하세요. 사용 가능: " + ", ".join(map(str, available))
        idx = simpledialog.askinteger("Camera Select", msg, initialvalue=available[0])
        root.destroy()
        return idx if idx in available else available[0]

    def _init_camera(self, index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            return cap
        return None

    def _set_prop_checked(self, prop_id, value, retries=2):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ok = False
        read_val = None
        for _ in range(retries):
            ok = bool(self.cap.set(prop_id, value))
            time.sleep(0.03)
            read_val = self.cap.get(prop_id)
            if ok:
                break
        return ok, read_val

    def read_camera_params(self):
        if self.cap is None or not self.cap.isOpened():
            return {}
        return {
            "gain": float(self.cap.get(cv2.CAP_PROP_GAIN)),
            "exposure": float(self.cap.get(cv2.CAP_PROP_EXPOSURE)),
            "brightness": float(self.cap.get(cv2.CAP_PROP_BRIGHTNESS)),
            "contrast": float(self.cap.get(cv2.CAP_PROP_CONTRAST)),
            "saturation": float(self.cap.get(cv2.CAP_PROP_SATURATION)),
            "wb_temperature": float(self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)),
            "auto_exposure": float(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "auto_wb": float(self.cap.get(cv2.CAP_PROP_AUTO_WB)),
        }

    def apply_camera_params(self, save=True):
        if self.cap is None or not self.cap.isOpened():
            return

        p = self.cam_params
        auto_exposure_val = 0.75 if p.get("auto_exposure", False) else 0.25
        auto_wb_val = 1 if p.get("auto_wb", False) else 0

        self._set_prop_checked(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_val)
        if not p.get("auto_exposure", False):
            self._set_prop_checked(cv2.CAP_PROP_EXPOSURE, float(p.get("exposure", -6.0)), retries=3)

        self._set_prop_checked(cv2.CAP_PROP_GAIN, float(p.get("gain", 0.0)), retries=3)
        self._set_prop_checked(cv2.CAP_PROP_BRIGHTNESS, float(p.get("brightness", 128.0)), retries=3)
        self._set_prop_checked(cv2.CAP_PROP_CONTRAST, float(p.get("contrast", 32.0)), retries=3)
        self._set_prop_checked(cv2.CAP_PROP_SATURATION, float(p.get("saturation", 64.0)), retries=3)

        self._set_prop_checked(cv2.CAP_PROP_AUTO_WB, auto_wb_val)
        if not p.get("auto_wb", False):
            self._set_prop_checked(cv2.CAP_PROP_WB_TEMPERATURE, float(p.get("wb_temperature", 4500.0)), retries=3)

        self.gain_val = float(p.get("gain", 0.0))
        if save:
            self.save_config()

    def reset_camera_params_to_default(self, save=True):
        self.cam_params.update({
            "gain": 0.0,
            "exposure": -6.0,
            "brightness": 128.0,
            "contrast": 32.0,
            "saturation": 64.0,
            "wb_temperature": 4500.0,
            "auto_exposure": False,
            "auto_wb": True,
        })
        self.apply_camera_params(save=save)

    def open_camera_param_window(self):
        if self.cam_param_ui is not None:
            try:
                self.cam_param_ui["root"].deiconify()
                self.cam_param_ui["root"].lift()
                return
            except tk.TclError:
                self.cam_param_ui = None

        if self.cap is None or not self.cap.isOpened():
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("Camera", "카메라가 연결되지 않았습니다.")
            root.destroy()
            return

        initial_params = dict(self.cam_params)

        root = tk.Tk()
        root.title("카메라 수동 파라미터")
        root.resizable(False, False)

        vars_map = {
            "gain": tk.DoubleVar(value=float(self.cam_params.get("gain", 0.0))),
            "exposure": tk.DoubleVar(value=float(self.cam_params.get("exposure", -6.0))),
            "brightness": tk.DoubleVar(value=float(self.cam_params.get("brightness", 128.0))),
            "contrast": tk.DoubleVar(value=float(self.cam_params.get("contrast", 32.0))),
            "saturation": tk.DoubleVar(value=float(self.cam_params.get("saturation", 64.0))),
            "wb_temperature": tk.DoubleVar(value=float(self.cam_params.get("wb_temperature", 4500.0))),
            "auto_exposure": tk.BooleanVar(value=bool(self.cam_params.get("auto_exposure", False))),
            "auto_wb": tk.BooleanVar(value=bool(self.cam_params.get("auto_wb", False))),
        }
        scale_widgets = []

        read_vars = {
            "gain": tk.StringVar(value="-"),
            "exposure": tk.StringVar(value="-"),
            "brightness": tk.StringVar(value="-"),
            "contrast": tk.StringVar(value="-"),
            "saturation": tk.StringVar(value="-"),
            "wb_temperature": tk.StringVar(value="-"),
            "auto_exposure": tk.StringVar(value="-"),
            "auto_wb": tk.StringVar(value="-"),
        }

        row = 0

        def add_scale(label, key, frm, to, res=1.0):
            nonlocal row
            tk.Label(root, text=label, anchor="w", width=16).grid(row=row, column=0, padx=8, pady=4, sticky="w")
            scale = tk.Scale(root, from_=frm, to=to, orient="horizontal", resolution=res,
                             variable=vars_map[key], length=280)
            scale.grid(row=row, column=1, padx=8, pady=4, sticky="ew")
            scale_widgets.append(scale)
            row += 1
            return scale

        def collect_and_apply(save_flag):
            self.cam_params.update({
                "gain": float(vars_map["gain"].get()),
                "exposure": float(vars_map["exposure"].get()),
                "brightness": float(vars_map["brightness"].get()),
                "contrast": float(vars_map["contrast"].get()),
                "saturation": float(vars_map["saturation"].get()),
                "wb_temperature": float(vars_map["wb_temperature"].get()),
                "auto_exposure": bool(vars_map["auto_exposure"].get()),
                "auto_wb": bool(vars_map["auto_wb"].get()),
            })
            self.apply_camera_params(save=save_flag)

        add_scale("Gain", "gain", 0, 128, 1)
        add_scale("Exposure", "exposure", -13, -1, 0.1)
        add_scale("Brightness", "brightness", 0, 255, 1)
        add_scale("Contrast", "contrast", 0, 127, 1)
        add_scale("Saturation", "saturation", 0, 255, 1)
        add_scale("WB Temperature", "wb_temperature", 2800, 6500, 50)

        auto_exp_chk = tk.Checkbutton(root, text="Auto Exposure", variable=vars_map["auto_exposure"])
        auto_exp_chk.grid(row=row, column=0, padx=8, pady=4, sticky="w")
        auto_wb_chk = tk.Checkbutton(root, text="Auto White Balance", variable=vars_map["auto_wb"])
        auto_wb_chk.grid(row=row, column=1, padx=8, pady=4, sticky="w")
        row += 1

        read_frame = tk.LabelFrame(root, text="현재 카메라 읽기값")
        read_frame.grid(row=row, column=0, columnspan=2, padx=8, pady=6, sticky="ew")

        tk.Label(read_frame, text="Gain", width=14, anchor="w").grid(row=0, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["gain"], width=18, anchor="w").grid(row=0, column=1, padx=6, pady=2, sticky="w")

        tk.Label(read_frame, text="Exposure", width=14, anchor="w").grid(row=1, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["exposure"], width=18, anchor="w").grid(row=1, column=1, padx=6, pady=2, sticky="w")

        tk.Label(read_frame, text="Brightness", width=14, anchor="w").grid(row=2, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["brightness"], width=18, anchor="w").grid(row=2, column=1, padx=6, pady=2, sticky="w")

        tk.Label(read_frame, text="Contrast", width=14, anchor="w").grid(row=3, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["contrast"], width=18, anchor="w").grid(row=3, column=1, padx=6, pady=2, sticky="w")

        tk.Label(read_frame, text="Saturation", width=14, anchor="w").grid(row=4, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["saturation"], width=18, anchor="w").grid(row=4, column=1, padx=6, pady=2, sticky="w")

        tk.Label(read_frame, text="WB Temp", width=14, anchor="w").grid(row=5, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["wb_temperature"], width=18, anchor="w").grid(row=5, column=1, padx=6, pady=2, sticky="w")

        tk.Label(read_frame, text="Auto Exposure", width=14, anchor="w").grid(row=6, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["auto_exposure"], width=18, anchor="w").grid(row=6, column=1, padx=6, pady=2, sticky="w")

        tk.Label(read_frame, text="Auto WB", width=14, anchor="w").grid(row=7, column=0, padx=6, pady=2, sticky="w")
        tk.Label(read_frame, textvariable=read_vars["auto_wb"], width=18, anchor="w").grid(row=7, column=1, padx=6, pady=2, sticky="w")

        row += 1

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)

        def refresh_read_values():
            values = self.read_camera_params()
            if not values:
                for key in read_vars:
                    read_vars[key].set("N/A")
                return

            read_vars["gain"].set(f"{values['gain']:.2f}")
            read_vars["exposure"].set(f"{values['exposure']:.2f}")
            read_vars["brightness"].set(f"{values['brightness']:.2f}")
            read_vars["contrast"].set(f"{values['contrast']:.2f}")
            read_vars["saturation"].set(f"{values['saturation']:.2f}")
            read_vars["wb_temperature"].set(f"{values['wb_temperature']:.2f}")
            read_vars["auto_exposure"].set(f"{values['auto_exposure']:.2f}")
            read_vars["auto_wb"].set(f"{values['auto_wb']:.2f}")

        def on_live_change(_=None):
            collect_and_apply(save_flag=False)
            refresh_read_values()

        for widget in scale_widgets:
            widget.configure(command=on_live_change)
        auto_exp_chk.configure(command=on_live_change)
        auto_wb_chk.configure(command=on_live_change)

        def on_preview():
            collect_and_apply(save_flag=False)
            refresh_read_values()

        def on_save():
            collect_and_apply(save_flag=True)
            root.destroy()

        def on_cancel():
            self.cam_params = dict(initial_params)
            self.apply_camera_params(save=False)
            root.destroy()

        def on_reset_default():
            self.reset_camera_params_to_default(save=False)
            vars_map["gain"].set(self.cam_params["gain"])
            vars_map["exposure"].set(self.cam_params["exposure"])
            vars_map["brightness"].set(self.cam_params["brightness"])
            vars_map["contrast"].set(self.cam_params["contrast"])
            vars_map["saturation"].set(self.cam_params["saturation"])
            vars_map["wb_temperature"].set(self.cam_params["wb_temperature"])
            vars_map["auto_exposure"].set(self.cam_params["auto_exposure"])
            vars_map["auto_wb"].set(self.cam_params["auto_wb"])
            refresh_read_values()

        def on_native_camera_setting():
            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

        tk.Button(btn_frame, text="미리 적용", width=10, command=on_preview).pack(side="left", padx=4)
        tk.Button(btn_frame, text="값 읽기", width=10, command=refresh_read_values).pack(side="left", padx=4)
        tk.Button(btn_frame, text="기본값 복원", width=10, command=on_reset_default).pack(side="left", padx=4)
        tk.Button(btn_frame, text="카메라 설정", width=10, command=on_native_camera_setting).pack(side="left", padx=4)
        tk.Button(btn_frame, text="저장", width=10, command=on_save).pack(side="left", padx=4)
        tk.Button(btn_frame, text="취소", width=10, command=on_cancel).pack(side="left", padx=4)

        refresh_read_values()
        def on_close_window():
            try:
                on_cancel()
            finally:
                self.cam_param_ui = None

        root.protocol("WM_DELETE_WINDOW", on_close_window)
        self.cam_param_ui = {"root": root, "refresh": refresh_read_values}

    def _process_camera_param_window(self):
        if self.cam_param_ui is None:
            return
        root = self.cam_param_ui.get("root")
        if root is None:
            self.cam_param_ui = None
            return
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            self.cam_param_ui = None

    def get_match_score(self, img_m_roi, img_t_roi, r_type):
        try:
            lab_m = cv2.cvtColor(img_m_roi, cv2.COLOR_BGR2LAB)
            lab_t = cv2.cvtColor(img_t_roi, cv2.COLOR_BGR2LAB)
            gray_m = lab_m[:, :, 0]
            gray_t = lab_t[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_m = clahe.apply(gray_m)
            gray_t = clahe.apply(gray_t)

            h, w = gray_m.shape[:2]
            if h < 10 or w < 10:
                return 0

            pad = 3
            search = cv2.copyMakeBorder(gray_t, pad, pad, pad, pad, cv2.BORDER_REFLECT)
            res_align = cv2.matchTemplate(search, gray_m, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res_align)
            gray_t = search[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]

            m_mean, m_std = cv2.meanStdDev(gray_m)
            t_mean, t_std = cv2.meanStdDev(gray_t)
            m_std = max(1.0, float(m_std[0][0]))
            t_std = max(1.0, float(t_std[0][0]))
            gray_t = np.clip(((gray_t - float(t_mean[0][0])) * (m_std / t_std)) + float(m_mean[0][0]), 0, 255).astype(np.uint8)

            bg_m = cv2.GaussianBlur(gray_m, (0, 0), 7)
            bg_t = cv2.GaussianBlur(gray_t, (0, 0), 7)
            bg_m = np.maximum(bg_m, 1)
            bg_t = np.maximum(bg_t, 1)
            gray_m = cv2.divide(gray_m, bg_m, scale=128)
            gray_t = cv2.divide(gray_t, bg_t, scale=128)

            blur_m = cv2.GaussianBlur(gray_m, (5, 5), 0)
            blur_t = cv2.GaussianBlur(gray_t, (5, 5), 0)

            h, w = blur_m.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            if r_type == 0:
                inset = max(2, int(min(w, h) * 0.15))
                cv2.rectangle(mask, (0, 0), (w-1, h-1), 255, -1)
                cv2.rectangle(mask, (inset, inset), (w-1-inset, h-1-inset), 0, -1)
            else:
                outer_r = min(w, h) // 2
                inner_r = max(2, int(outer_r * 0.7))
                cv2.circle(mask, (w//2, h//2), outer_r, 255, -1)
                cv2.circle(mask, (w//2, h//2), inner_r, 0, -1)

            hp_m = cv2.Laplacian(blur_m, cv2.CV_32F, ksize=3)
            hp_t = cv2.Laplacian(blur_t, cv2.CV_32F, ksize=3)
            idx = mask > 0
            if not np.any(idx):
                return 0
            vec_m = hp_m[idx].astype(np.float32)
            vec_t = hp_t[idx].astype(np.float32)
            norm_m = float(np.linalg.norm(vec_m))
            norm_t = float(np.linalg.norm(vec_t))
            if norm_m < 1e-6 or norm_t < 1e-6:
                pixel_match = 100.0
            else:
                cos_sim = float(np.dot(vec_m, vec_t) / (norm_m * norm_t))
                cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                pixel_match = ((cos_sim + 1.0) * 0.5) * 100.0

            med = np.median(blur_m)
            low = int(max(20, 0.66 * med))
            high = int(min(255, max(low + 20, 1.33 * med)))
            edges_m = cv2.Canny(blur_m, low, high)
            edges_t = cv2.Canny(blur_t, low, high)
            edges_m = cv2.bitwise_and(edges_m, mask)
            edges_t = cv2.bitwise_and(edges_t, mask)
            edge_inter = cv2.countNonZero(cv2.bitwise_and(edges_m, edges_t))
            edge_union = cv2.countNonZero(cv2.bitwise_or(edges_m, edges_t))
            edge_match = 100.0 if edge_union == 0 else (edge_inter / edge_union) * 100.0

            match = (pixel_match * 0.4) + (edge_match * 0.6)
            return match
        except:
            return 0

    def save_ng_image(self, frame):
        now = datetime.now()
        if (now.timestamp() - self.last_ng_time) > 3:
            filename = now.strftime("NG_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(self.log_path, filename), frame)
            self.last_ng_time = now.timestamp()

    def _append_result_csv(self, result_text):
        is_new = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "result"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result_text])

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
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.setMouseCallback(win_name, self.on_mouse)

        while True:
            self._process_camera_param_window()
            ret, frame_raw = self.cap.read()
            if not ret: break
            self.frame_buffer.append(frame_raw)
            if len(self.frame_buffer) > 1:
                frame_raw = np.mean(np.stack(self.frame_buffer, axis=0), axis=0).astype(np.uint8)
            h_m, w_m = self.img_m.shape[:2]
            aspect = frame_raw.shape[1] / frame_raw.shape[0]
            nw, nh = w_m, int(w_m / aspect)
            resized_cam = cv2.resize(frame_raw, (nw, nh))
            canvas_cam = np.zeros((h_m, w_m, 3), dtype=np.uint8)
            yo, xo = (h_m - nh) // 2, (w_m - nw) // 2
            canvas_cam[yo:yo+nh, xo:xo+nw] = resized_cam; self.last_canvas = canvas_cam.copy()

            main_view = np.hstack((self.img_m.copy(), canvas_cam))
            status_bar = np.zeros((40, main_view.shape[1], 3), dtype=np.uint8)
            cv2.putText(status_bar, f"MATCH-TH: {self.th_match}% | LOG: {self.log_path}", (20, 27), 0, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            if self.last_result and (time.time() - self.last_result_time) <= self.result_hold_sec:
                r_text, r_color = self.last_result
                cv2.putText(status_bar, f"RESULT: {r_text}", (980, 27), 0, 0.5, r_color, 2, cv2.LINE_AA)
            main_view = np.vstack((status_bar, main_view))

            # 1. 드래그 중일 때 참조라인 실시간 표시
            if self.is_dragging:
                sx, sy = self.start_point; cx, cy = self.curr_point
                if self.draw_mode == 0:
                    cv2.rectangle(main_view, (sx, sy+40), (cx, cy+40), self.CLR['guide'], 1)
                else:
                    r = int(np.sqrt((sx-cx)**2 + (sy-cy)**2))
                    cv2.circle(main_view, (sx, sy+40), r, self.CLR['guide'], 1)

            do_inspect = self.inspection_mode or self.inspect_triggered
            inspect_result_text = None
            has_ng = False
            for i, (rx, ry, rw, rh, rt) in enumerate(self.roi_list):
                oy = 40
                if rt == 0: cv2.rectangle(main_view, (rx, ry+oy), (rx+rw, ry+rh+oy), self.CLR['guide'], 1)
                else: cv2.circle(main_view, (rx+rw//2, ry+rh//2+oy), rw//2, self.CLR['guide'], 1)
                
                if do_inspect:
                    tx, ty = self._track_roi(i, canvas_cam, rx, ry, rw, rh)
                    m_roi = self.img_m[ry:ry+rh, rx:rx+rw]
                    t_roi = canvas_cam[ty:ty+rh, tx:tx+rw]
                    if m_roi.size == 0 or t_roi.size == 0 or m_roi.shape[:2] != t_roi.shape[:2]:
                        continue
                    
                    s_match = self.get_match_score(m_roi, t_roi, rt)
                    is_ok = s_match >= self.th_match
                    if not is_ok: has_ng = True

                    ex, ey = tx + w_m, ty + oy
                    color = self.CLR['ok'] if is_ok else self.CLR['ng']
                    if rt == 0: cv2.rectangle(main_view, (ex, ey), (ex+rw, ey+rh), color, 2)
                    else: cv2.circle(main_view, (ex+rw//2, ey+rh//2), rw//2, color, 2)
                    
                    cv2.putText(main_view, f"MATCH:{s_match:.0f}%", (ex, ey-10), 0, 0.35, color, 1)

            if self.inspect_triggered and self.roi_list:
                result_text = "NG" if has_ng else "OK"
                result_color = self.CLR['ng'] if has_ng else self.CLR['ok']
                self.last_result = (result_text, result_color)
                self.last_result_time = time.time()
                inspect_result_text = result_text
            elif self.inspect_triggered and not self.roi_list:
                self.last_result = ("NO ROI", self.CLR['warn'])
                self.last_result_time = time.time()
                inspect_result_text = "NO ROI"

            if self.inspection_mode and has_ng: self.save_ng_image(canvas_cam)

            btn_bar = np.zeros((80, main_view.shape[1], 3), dtype=np.uint8)
            btn_bar[:] = self.CLR['bg']
            self._draw_button(btn_bar, 10, 80, "RESET", self.CLR['btn_reset'])
            self._draw_button(btn_bar, 90, 160, "RECT", (180, 90, 0), active=(self.draw_mode==0))
            self._draw_button(btn_bar, 170, 240, "CIRCLE", (0, 130, 180), active=(self.draw_mode==1))
            self._draw_button(btn_bar, 250, 360, "SET MASTER", self.CLR['master'])
            self._draw_button(btn_bar, 370, 480, "MATCH TH", self.CLR['thresh'])
            self._draw_button(btn_bar, 490, 620, "CAM PARAM", (120, 80, 180))
            self._draw_button(btn_bar, 630, 740, "INSPECT", (80, 120, 200))
            self._draw_button(btn_bar, 750, 850, "START", self.CLR['btn_active'] if self.inspection_mode else (70, 70, 70))
            self._draw_button(btn_bar, main_view.shape[1]-100, main_view.shape[1]-10, "EXIT", (50, 50, 50))

            if self.inspect_triggered:
                if inspect_result_text:
                    self._append_result_csv(inspect_result_text)
                self.inspect_triggered = False
            
            base_frame = np.vstack((main_view, btn_bar))
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
            except:
                win_w, win_h = base_frame.shape[1], base_frame.shape[0]
            if win_w > 0 and win_h > 0 and (win_w != base_frame.shape[1] or win_h != base_frame.shape[0]):
                self.display_scale = (win_w / base_frame.shape[1], win_h / base_frame.shape[0])
                disp_frame = cv2.resize(base_frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
            else:
                self.display_scale = (1.0, 1.0)
                disp_frame = base_frame

            cv2.imshow(win_name, disp_frame)
            if cv2.waitKey(1) == 27: break
        self.cap.release(); cv2.destroyAllWindows()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.th_match = data.get("th_match", 90.0)
                self.roi_list = [tuple(r) for r in data.get("roi_list", [])]
                self.gain_val = data.get("gain", 0)
                loaded_cam = data.get("cam_params", {})
                self.cam_params.update({
                    "gain": float(loaded_cam.get("gain", self.gain_val)),
                    "exposure": float(loaded_cam.get("exposure", -6.0)),
                    "brightness": float(loaded_cam.get("brightness", 128.0)),
                    "contrast": float(loaded_cam.get("contrast", 32.0)),
                    "saturation": float(loaded_cam.get("saturation", 64.0)),
                    "wb_temperature": float(loaded_cam.get("wb_temperature", 4500.0)),
                    "auto_exposure": bool(loaded_cam.get("auto_exposure", False)),
                    "auto_wb": bool(loaded_cam.get("auto_wb", True)),
                })
        else:
            self.cam_params["gain"] = float(self.gain_val)

    def save_config(self):
        data = {
            "th_match": self.th_match,
            "roi_list": self.roi_list,
            "gain": self.gain_val,
            "cam_params": self.cam_params,
        }
        with open(self.config_file, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)

    def _draw_button(self, bar, x1, x2, text, color, active=False):
        cv2.rectangle(bar, (x1, 10), (x2, 60), color, -1, cv2.LINE_AA)
        if active: cv2.rectangle(bar, (x1, 10), (x2, 60), (255, 255, 255), 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(text, 0, 0.35, 1)
        cv2.putText(bar, text, (x1 + (x2-x1-tw)//2, 40), 0, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    def on_mouse(self, event, x, y, flags, param):
        sx = int(x / self.display_scale[0]) if self.display_scale[0] > 0 else x
        sy = int(y / self.display_scale[1]) if self.display_scale[1] > 0 else y
        ry = sy - 40
        if event == cv2.EVENT_LBUTTONDOWN:
            if sy > self.img_m.shape[0] + 40: self._handle_btns(sx)
            elif sy > 40 and sx < self.img_m.shape[1]:
                self.is_dragging, self.start_point = True, (sx, ry); self.curr_point = (sx, ry)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging: self.curr_point = (sx, ry)
        elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
            self.is_dragging = False
            start_x, start_y = self.start_point
            end_x, end_y = sx, ry
            if self.draw_mode == 0:
                rw, rh = abs(end_x-start_x), abs(end_y-start_y)
                if rw > 10:
                    self.roi_list.append((min(start_x, end_x), min(start_y, end_y), rw, rh, 0))
                    self._reset_tracking()
                    self.save_config()
            else:
                radius = int(np.sqrt((start_x-end_x)**2 + (start_y-end_y)**2))
                if radius > 5:
                    self.roi_list.append((start_x-radius, start_y-radius, radius*2, radius*2, 1))
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
            s = simpledialog.askfloat("MATCH-TH", "일치율 임계치(%)", initialvalue=self.th_match)
            if s: self.th_match = s; self.save_config()
            root.destroy()
        elif 490 <= x <= 620:
            self.open_camera_param_window()
        elif 630 <= x <= 740:
            self.inspect_triggered = True
        elif 750 <= x <= 850:
            self.inspection_mode = not self.inspection_mode if self.roi_list else False
            if self.inspection_mode:
                self._reset_tracking()
        elif x > (self.img_m.shape[1] * 2 - 100): sys.exit()

if __name__ == "__main__":
    inspector = VisionInspector(r'C:\VisionMaster\master.png')
    inspector.run()