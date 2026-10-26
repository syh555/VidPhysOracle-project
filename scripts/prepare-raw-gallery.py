from pathlib import Path
import json,cv2
r=Path(__file__).resolve().parents[1]
data=Path('/Users/suyihan/personal_project/worldmodelbench')
part=data/'part_of_raw';supp=data/'Supplementary/Raw';raw=data/'Raw_24_samples/Raw'
specs=[
 ('rigid-falling_objects','Falling objects','Rigid bodies',[part/'falling_objects_0098/render.mp4',part/'falling_objects_0115/render.mp4']),
 ('rigid-rolling_objects','Rolling objects','Rigid bodies',[part/'rolling_objects_0037/render.mp4',part/'rolling_objects_0041/render.mp4']),
 ('rigid-dominoes','Dominoes','Rigid bodies',[part/'dominoes_0000/render.mp4',part/'dominoes_0119/render.mp4']),
 ('soft-falling_objects_on_soft_pad','Soft pad','Soft bodies',[part/'falling_objects_on_soft_pad_0029/render.mp4',part/'falling_objects_on_soft_pad_0107/render.mp4']),
 ('soft-gravity_draping_flag','Cloth','Soft bodies',[part/'gravity_draping_flag_0033/render.mp4',part/'gravity_draping_flag_0123/render.mp4']),
 ('soft-gravity_tightening_soft_rope','Rope','Soft bodies',[part/'gravity_tightening_soft_rope_0041/render.mp4',part/'gravity_tightening_soft_rope_0086/render.mp4']),
 ('fluid-rigid_objects_in_water_tank','Water tank','Liquids',[part/'rigid_objects_in_water_tank_0012/render.mp4',part/'rigid_objects_in_water_tank_0030/render.mp4']),
 ('fluid-draining_tank','Draining tank','Liquids',[part/'draining_tank_0010/render.mp4',part/'draining_tank_0013/render.mp4']),
 ('fluid-u_connector','U connector','Liquids',[part/'u_connector_0012/render.mp4',part/'u_connector_0031/render.mp4']),
 ('gas-gas_smoke_stack','Smoke stack','Gases',[part/'gas_smoke_stack_0012/render.mp4',part/'gas_smoke_stack_0031/render.mp4']),
 ('gas-impact_smoke_dispersion','Impact smoke','Gases',[part/'impact_smoke_dispersion_0012/render.mp4',part/'impact_smoke_dispersion_0031/render.mp4']),
 ('gas-gas_flame_heating','Flame heating','Gases',[supp/'gas-gas_flame_heating/gas_flame_heating_0000/render.mp4',supp/'gas-gas_flame_heating/gas_flame_heating_0001/render.mp4']),
]
result=[]
for key,label,material,sources in specs:
 if not sources:
  sources=sorted((raw/key).glob('*/render.mp4'))[:2]
 group={'id':key,'label':label,'material':material,'videos':[]}
 for source in sources:
  cap=cv2.VideoCapture(str(source));ok,frame=cap.read()
  if not ok: cap.release();continue
  fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert fps>0
  size=(1280,720)
  stem=source.parent.name
  folder=r/'public/videos/raw'/key;folder.mkdir(parents=True,exist_ok=True)
  target=folder/f'{stem}.mp4';poster=folder/f'{stem}.jpg'
  writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'avc1'),fps,size);assert writer.isOpened()
  count=0
  while ok:
   resized=cv2.resize(frame,size,interpolation=cv2.INTER_AREA);writer.write(resized)
   if count==0: cv2.imwrite(str(poster),resized,[cv2.IMWRITE_JPEG_QUALITY,90])
   count+=1;ok,frame=cap.read()
  writer.release();cap.release();assert count==total
  group['videos'].append({'src':str(target.relative_to(r/'public')),'poster':str(poster.relative_to(r/'public')),'sample':stem,'source':str(source),'frames':count,'fps':fps})
 if group['videos']:result.append(group)
 print(label,len(group['videos']),flush=True)
(r/'src/data/raw-gallery.json').write_text(json.dumps(result,indent=2)+'\n')
