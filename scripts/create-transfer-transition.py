"""Create a 48-frame preview: raw frames 0-9, matched transfer frames 10-47.
Run with a Python environment containing OpenCV and Pillow.
"""
from pathlib import Path
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

root = Path(__file__).resolve().parents[1]
source_root = Path('/Users/suyihan/personal_project/worldmodelbench/part_of_raw')
scenes = json.loads((root / 'src/data/selected-transfer.json').read_text())
cases = []
result_scenes = []
for scene in scenes:
    result = {**scene, 'videos': []}
    for video in scene['videos']:
        stem = Path(video['filename']).stem
        source_id = stem.split('_appearance_')[0] if '_appearance_' in stem else stem.rsplit('_', 2)[0]
        if not (source_root / source_id / 'render.mp4').is_file():
            print('Skipped: no matching raw for', stem)
            continue
        cases.append((stem, source_id, str(Path(video['src']).relative_to('videos/transfer'))))
        result['videos'].append({**video,
            'src': f'videos/transition-demo/{stem}-raw10-transfer.mp4',
            'poster': f'videos/transition-demo/{stem}-poster.jpg'})
    if result['videos']:
        result_scenes.append(result)
for name, source_id, selected_video in cases:
    raw = source_root / source_id / 'render.mp4'
    transfer = root / 'public/videos/transfer' / selected_video
    out = root / 'public/videos/transition-demo'
    out.mkdir(parents=True, exist_ok=True)
    raw_cap, transfer_cap = cv2.VideoCapture(str(raw)), cv2.VideoCapture(str(transfer))
    fps = transfer_cap.get(cv2.CAP_PROP_FPS)
    assert fps == raw_cap.get(cv2.CAP_PROP_FPS) == 24
    assert raw_cap.get(cv2.CAP_PROP_FRAME_COUNT) == transfer_cap.get(cv2.CAP_PROP_FRAME_COUNT) == 48
    width, height = int(transfer_cap.get(3)), int(transfer_cap.get(4))
    writer = cv2.VideoWriter(str(out / f'{name}-raw10-transfer.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    assert writer.isOpened()
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 25)
    for i in range(48):
        ok_raw, raw_frame = raw_cap.read()
        ok_transfer, transfer_frame = transfer_cap.read()
        assert ok_raw and ok_transfer
        frame = cv2.resize(raw_frame, (width, height), interpolation=cv2.INTER_AREA) if i < 10 else transfer_frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        label = 'RAW' if i < 10 else 'TRANSFER'
        color = '#173f65' if i < 10 else '#16745a'
        draw.rounded_rectangle((22, 20, 215, 66), radius=10, fill=color)
        draw.text((38, 28), label, font=font, fill='white')
        writer.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if i == 0:
            image.save(out / f'{name}-poster.jpg', quality=90)
        if i in [8, 9, 10, 11]:
            image.save(root / 'output' / f'{name}-transition-frame-{i:02d}.jpg', quality=90)
    writer.release()
    raw_cap.release()
    transfer_cap.release()
    check = cv2.VideoCapture(str(out / f'{name}-raw10-transfer.mp4'))
    assert check.get(cv2.CAP_PROP_FRAME_COUNT) == 48 and check.get(cv2.CAP_PROP_FPS) == 24
    check.release()
    (out / f'{name}-manifest.json').write_text(json.dumps({
        'raw_source': str(raw), 'transfer_source': str(transfer),
        'raw_frames_zero_based': [0, 9], 'transfer_frames_zero_based': [10, 47],
        'fps': 24, 'frame_count': 48, 'switch_time_seconds': 10 / 24,
        'width': width, 'height': height,
        'spatial_processing': 'Raw resized to transfer dimensions; no motion interpolation, time warping, or crossfade.'
    }, indent=2) + '\n')
    print('Created 48-frame H264 preview; raw [0,9], transfer [10,47], switch at 0.4167s.')
(root / 'src/data/transition-transfer.json').write_text(json.dumps(result_scenes, indent=2) + '\n')
