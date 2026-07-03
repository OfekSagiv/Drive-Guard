"""
Path 1d at dataset scale — RGB GroundingDINO -> hand-association filter -> joint
homography warp -> IR labels (full-frame coords). Builds a YOLO dataset of full IR
frames + transferred labels, plus QC artifacts. Does NOT train. Diagnostic.

Per manifest frame (IR-frame index f in run R):
  - timestamp-sync: IR f -> RGB g, -> openpose row (nearest timestamp).
  - project 3D joints into IR (ids_2 calibration) -> IR 2D joints.
  - run pose on the bright RGB -> RGB 2D joints; homography RGB->IR from shared joints.
  - GroundingDINO on RGB -> object boxes; keep those near a wrist (held-object filter);
    warp kept boxes into IR via the homography.
  - write full IR frame + YOLO labels (normalized to the IR frame).

Run: python ir_object_detection/build_rgb_transfer.py
"""
import os, sys, json, bisect, random, collections
import cv2, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
BASE = os.environ.get("DA_3D_BASE", "/Users/ofeksagiv/Downloads")  # openpose_3d / interior_3d live here
DEV = "mps"
OUT = os.path.join(C.PKG_DIR, "rgb_transfer")

COCO = {0:'nose',1:'lEye',2:'rEye',3:'lEar',4:'rEar',5:'lShoulder',6:'rShoulder',7:'lElbow',
        8:'rElbow',9:'lWrist',10:'rWrist',11:'lHip',12:'rHip',13:'lKnee',14:'rKnee',15:'lAnkle',16:'rAnkle'}
VIZ = {'phone':(0,0,255),'drink':(0,255,0),'food':(0,255,255)}


def _ts(p): return [float(x) for x in open(p)]
def _near(arr, t):
    i = bisect.bisect_left(arr, t); c = [j for j in (i-1,i,i+1) if 0 <= j < len(arr)]
    return min(c, key=lambda j: abs(arr[j]-t))


def _ir_calib(vp, run):
    cal = json.load(open(f'data/a_column_co_driver/{vp}/{run}.calibration.json'))
    K = cal['intrinsics']; q = cal['extrinsics']['rotation']; t = cal['extrinsics']['translation']
    w,x,y,z = q['w'],q['x'],q['y'],q['z']
    R = np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                  [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                  [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    rvec = cv2.Rodrigues(R)[0]; tvec = np.array([t['x'],t['y'],t['z']], float)
    Km = np.array([[K['focallength']['fx'],0,K['principal_point']['cx']],
                   [0,K['focallength']['fy'],K['principal_point']['cy']],[0,0,1]], float)
    d = K['distortion']; dist = np.array([d['k1'],d['k2'],d['p1'],d['p2'],d['k3']], float)
    return rvec, tvec, Km, dist


def main():
    os.chdir(C.DATA_ROOT)
    for sub in ('images','labels','qc/random','qc/per_class','qc/failures'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    from ultralytics import YOLO
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from PIL import Image
    pose = YOLO(os.path.join(C.DATA_ROOT, C.POSE_WEIGHTS))
    proc = AutoProcessor.from_pretrained(C.GROUNDING_MODEL)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(C.GROUNDING_MODEL).to(DEV).eval()

    def gd_detect(bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = proc(images=pil, text=C.GROUNDING_PROMPT, return_tensors='pt').to(DEV)
        with torch.no_grad(): out = gd(**inp)
        r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.25,
              text_threshold=0.25, target_sizes=[pil.size[::-1]])[0]
        labs = r['text_labels'] if 'text_labels' in r else r['labels']
        res = []
        for b, s, l in zip(r['boxes'], r['scores'], labs):
            p = str(l).lower().strip(); cls=None; bl=0
            for k,c in C.PHRASE_TO_CLASS.items():
                if k in p and len(k) > bl: cls,bl = c,len(k)
            if cls: res.append((b.tolist(), float(s), cls))
        return res

    man = pd.read_csv(C.FROZEN_MANIFEST)
    man = man[man.file_id.apply(lambda f: os.path.exists(
        f"data/a_column_co_driver/{f.split('/')[0]}/{f.split('/')[1]}.calibration.json"))]
    runs = sorted(man.file_id.unique())
    rng = random.Random(0)

    stats = collections.defaultdict(int)
    cls_count = collections.Counter()
    boxes_per_img = []
    nobox_by_gt = collections.Counter()
    qc_pool = []          # (name, gt) for random/per-class QC
    fail_pool = []        # (name, reason)
    n_done = 0

    for fid in runs:
        vp, run = fid.split('/'); rb = run.replace('.ids_2','')
        try:
            ir_ts = _ts(f'data/a_column_co_driver/{vp}/{run}.timestamps')
            rgb_ts = _ts(f'data/codriver_color/{vp}/{rb}.kinect_color.timestamps')
            op = pd.read_csv(f'{BASE}/openpose_3d/{vp}/{rb}.ids_1.openpose.3d.csv')
            op_ts = list(op.timestamp.values)
            rvec, tvec, Km, dist = _ir_calib(vp, run)
        except Exception as e:
            print(f'skip run {fid}: {e}'); continue
        ic = cv2.VideoCapture(f'data/a_column_co_driver/{vp}/{run}.mp4')
        rc = cv2.VideoCapture(f'data/codriver_color/{vp}/{rb}.kinect_color.mp4')
        sub = man[man.file_id == fid]
        for _, row in sub.iterrows():
            n_done += 1
            if n_done % 100 == 0: print(f'  {n_done}/{len(man)} frames...', flush=True)
            f = int(row.frame); gt = row.obj_class; name = row.image
            T = ir_ts[f]; g = _near(rgb_ts, T); oprow = op.iloc[_near(op_ts, T)]
            ic.set(cv2.CAP_PROP_POS_FRAMES, f); ok1, ir = ic.read()
            rc.set(cv2.CAP_PROP_POS_FRAMES, g); ok2, rgb = rc.read()
            if not ok1 or not ok2: stats['read_fail'] += 1; continue
            H_, W_ = ir.shape[:2]
            # IR joints by projection
            names = [n for n in COCO.values() if f'{n}_x' in op.columns and oprow[f'{n}_p'] > 0.1]
            if len(names) < 4: stats['no_pose3d'] += 1; nobox_by_gt[gt] += 1; continue
            P3 = np.array([[oprow[f'{n}_x'],oprow[f'{n}_y'],oprow[f'{n}_z']] for n in names], float)
            ir_pts = cv2.projectPoints(P3, rvec, tvec, Km, dist)[0].reshape(-1,2)
            ir_jt = {n: ir_pts[k] for k,n in enumerate(names)}
            # RGB joints
            pr = pose(rgb, conf=0.25, verbose=False); rgb_jt = {}
            if pr and pr[0].keypoints is not None and len(pr[0].boxes) > 0:
                kp = pr[0].keypoints.data.cpu().numpy()[int(np.argmax(pr[0].boxes.conf.cpu().numpy()))]
                for idx,n in COCO.items():
                    if kp[idx,2] >= 0.3: rgb_jt[n] = kp[idx,:2]
            common = [n for n in ir_jt if n in rgb_jt]
            if len(common) < 4: stats['no_homography'] += 1; nobox_by_gt[gt] += 1; continue
            H, _ = cv2.findHomography(np.array([rgb_jt[n] for n in common],np.float32).reshape(-1,1,2),
                                      np.array([ir_jt[n] for n in common],np.float32).reshape(-1,1,2), cv2.RANSAC, 5.0)
            if H is None: stats['no_homography'] += 1; nobox_by_gt[gt] += 1; continue
            wrists = {w: rgb_jt[w] for w in ('lWrist','rWrist') if w in rgb_jt}
            lines = []; box_failed = False
            for (x1,y1,x2,y2), s, cls in gd_detect(rgb):
                cxb, cyb = (x1+x2)/2, (y1+y2)/2
                if not wrists: continue
                wn = min(wrists, key=lambda w: np.hypot(*(wrists[w]-[cxb,cyb])))
                if np.hypot(*(wrists[wn]-[cxb,cyb])) > max(150.0, 1.3*max(x2-x1, y2-y1)): continue
                corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32).reshape(-1,1,2)
                wc = cv2.perspectiveTransform(corners, H).reshape(-1,2)
                bx1,by1,bx2,by2 = wc[:,0].min(),wc[:,1].min(),wc[:,0].max(),wc[:,1].max()
                bw,bh = bx2-bx1, by2-by1
                # failure flags: out of frame / absurd size
                if bw > 0.7*W_ or bh > 0.7*H_ or bx2 < 0 or by2 < 0 or bx1 > W_ or by1 > H_:
                    box_failed = True; continue
                cx,cy = ((bx1+bx2)/2)/W_, ((by1+by2)/2)/H_
                lines.append(f"{C.CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {bw/W_:.6f} {bh/H_:.6f}")
                cls_count[cls] += 1
            cv2.imwrite(os.path.join(OUT,'images',name), ir, [int(cv2.IMWRITE_JPEG_QUALITY),92])
            if lines:
                open(os.path.join(OUT,'labels',name.replace(C.IMG_EXT,'.txt')),'w').write("\n".join(lines))
                boxes_per_img.append(len(lines)); stats['frames_with_labels'] += 1
                qc_pool.append((name, gt))
            else:
                if gt == 'safe':
                    open(os.path.join(OUT,'labels',name.replace(C.IMG_EXT,'.txt')),'w').write(""); stats['neg_safe'] += 1
                else:
                    nobox_by_gt[gt] += 1; stats['missed_object'] += 1
            if box_failed: fail_pool.append((name, gt))
        ic.release(); rc.release()

    # ---- QC overlays (re-render a sample as RGB|IR? here: IR with boxes from labels) ----
    def draw(name):
        ir = cv2.imread(os.path.join(OUT,'images',name));
        lp = os.path.join(OUT,'labels',name.replace(C.IMG_EXT,'.txt'))
        if ir is None or not os.path.exists(lp): return None
        H_,W_ = ir.shape[:2]
        for ln in open(lp):
            if not ln.strip(): continue
            ci,cx,cy,bw,bh = ln.split(); ci=int(ci); cx,cy,bw,bh=[float(v) for v in (cx,cy,bw,bh)]
            x1,y1 = int((cx-bw/2)*W_),int((cy-bh/2)*H_); x2,y2=int((cx+bw/2)*W_),int((cy+bh/2)*H_)
            col = VIZ[C.CLASSES[ci]]; cv2.rectangle(ir,(x1,y1),(x2,y2),col,2)
            cv2.putText(ir,C.CLASSES[ci],(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
        return ir
    rng.shuffle(qc_pool)
    for name,gt in qc_pool[:40]:
        im = draw(name);  im is not None and cv2.imwrite(os.path.join(OUT,'qc/random',name), im)
    bycls = collections.defaultdict(list)
    for name,gt in qc_pool: bycls[gt].append(name)
    for gt,names in bycls.items():
        for name in names[:10]:
            im = draw(name); im is not None and cv2.imwrite(os.path.join(OUT,'qc/per_class',f'{gt}__{name}'), im)
    for name,gt in fail_pool[:40]:
        im = draw(name); im is not None and cv2.imwrite(os.path.join(OUT,'qc/failures',f'{gt}__{name}'), im)

    summary = {
        'frames_total': int(len(man)),
        'frames_with_labels': stats['frames_with_labels'],
        'neg_safe': stats['neg_safe'],
        'missed_object': stats['missed_object'],
        'no_pose3d': stats['no_pose3d'], 'no_homography': stats['no_homography'], 'read_fail': stats['read_fail'],
        'total_boxes': sum(cls_count.values()),
        'class_distribution': dict(cls_count),
        'avg_boxes_per_labeled_img': round(np.mean(boxes_per_img),3) if boxes_per_img else 0,
        'no_box_by_gt': dict(nobox_by_gt),
        'box_failures_flagged': len(fail_pool),
    }
    json.dump(summary, open(os.path.join(OUT,'stats.json'),'w'), indent=2)
    print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2))
    print(f"\nDataset: {OUT}/images + /labels | QC: {OUT}/qc/{{random,per_class,failures}}")


if __name__ == "__main__":
    main()
