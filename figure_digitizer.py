from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def render_figure_digitizer() -> None:
    """Compact, beginner-friendly 2D figure digitizer.

    Client-side only. Extracted values are intentionally not persisted.
    """
    st.caption("그래프 이미지를 보정한 뒤 클릭해 Mean, SD/SE/95% CI 값을 읽습니다. 추출값은 저장되지 않습니다.")

    html = r'''
    <style>
      :root{--bg:#f7f9fc;--card:#fff;--line:#d9e0ea;--text:#172033;--muted:#667085;--accent:#2563eb;--ok:#14804a;--warn:#b45309;}
      *{box-sizing:border-box} body{margin:0;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:transparent}
      .app{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:12px;align-items:start}
      .top{grid-column:1/-1;display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:#fff}
      .steps{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.step{display:flex;gap:6px;align-items:center;padding:6px 9px;border-radius:999px;background:#f3f6fa;color:#667085;font-size:12px}.step b{width:20px;height:20px;border-radius:50%;display:grid;place-items:center;background:#e5eaf1;color:#475467}.step.active{background:#eef4ff;color:#174db1}.step.active b,.step.done b{background:var(--accent);color:#fff}.step.done{color:#344054}
      .upload{display:flex;gap:8px;align-items:center;min-width:300px}.upload input{font-size:12px;max-width:245px}.fileName{font-size:11px;color:var(--muted);max-width:170px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}.workspace{padding:0;overflow:hidden}
      .toolbar{padding:8px 10px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap;background:#fff}.btnrow{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
      button{border:1px solid #cfd8e5;background:#fff;border-radius:8px;padding:7px 9px;font-size:12px;cursor:pointer;color:var(--text)}button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:700}.btnrow button.active{background:#eaf1ff;border-color:#8db0ff;color:#174db1}button:disabled{opacity:.45;cursor:not-allowed}
      .zoomText{font-size:11px;color:var(--muted);min-width:46px;text-align:center}.canvasWrap{position:relative;background:#f3f5f8;height:680px;overflow:auto}.canvasWrap.empty{display:grid;place-items:center}.canvasWrap.empty:after{content:"위에서 그래프 이미지를 업로드하세요";color:#7b8798;font-size:14px}.stage{position:relative;transform-origin:top left;margin:14px}.stage canvas{display:block;max-width:none}.overlay{position:absolute;left:0;top:0;cursor:crosshair;touch-action:none}
      .statusbar{padding:8px 10px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;background:#fff}.instruction{color:#344054;font-weight:650}
      .mag{position:absolute;right:18px;bottom:18px;width:270px;height:270px;border:3px solid #fff;border-radius:12px;box-shadow:0 6px 24px rgba(0,0,0,.22);background:#fff;display:none;overflow:hidden;z-index:6;pointer-events:none}.mag canvas{width:100%;height:100%}.mag:before,.mag:after{content:"";position:absolute;background:#ef4444;z-index:2}.mag:before{width:1px;height:100%;left:50%;top:0}.mag:after{height:1px;width:100%;top:50%;left:0}.magLabel{position:absolute;left:8px;top:7px;z-index:3;background:rgba(255,255,255,.88);padding:3px 6px;border-radius:6px;font-size:11px;color:#344054}
      .side{display:grid;gap:10px}.title{font-size:14px;font-weight:750;margin:0 0 8px}.sub{font-size:11px;color:var(--muted);line-height:1.45}label{display:block;font-size:11px;font-weight:650;margin:8px 0 4px}input[type=number],select{width:100%;border:1px solid #cfd8e5;border-radius:8px;padding:7px 8px;background:#fff;color:var(--text);font-size:12px}.radio-row{display:flex;gap:10px;flex-wrap:wrap;font-size:11px}.radio-row label{font-weight:500;margin:0;display:flex;gap:4px;align-items:center}.cal-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.full{width:100%;margin-top:9px}.hint,.okbox,.warnbox{margin-top:8px;border-radius:8px;padding:8px;font-size:11px;line-height:1.45}.hint{background:#f7f9fc;color:var(--muted)}.okbox{border:1px solid #b9dfc8;background:#f2fbf5;color:#17623a}.warnbox{border:1px solid #f1d39b;background:#fffbeb;color:#8a5509}
      .precision{display:grid;grid-template-columns:1fr 1fr;gap:6px}.precision button{padding:7px}.precision button.active{background:#eef4ff;border-color:#8db0ff;color:#174db1}.results{margin-top:9px;border:1px solid #dce5f4;background:#f8fbff;border-radius:10px;padding:9px}.r{display:grid;grid-template-columns:1fr auto;gap:8px;padding:6px 0;border-top:1px solid #e5ecf6;font-size:12px}.r:first-child{border-top:0}.r b{font-size:14px}.mean{color:#1d4ed8}.err{color:#dc2626}.conv{color:#16803a}.small{font-size:10.5px;color:var(--muted);line-height:1.4}
      .compactHelp{padding:8px 10px;background:#f8fafc;border-top:1px solid var(--line);font-size:11px;color:#667085}.compactHelp b{color:#344054}
      @media(max-width:950px){.app{grid-template-columns:1fr}.side{grid-template-columns:1fr 1fr}.canvasWrap{height:590px}}
      @media(max-width:680px){.side{display:grid;grid-template-columns:1fr}.upload{min-width:0;width:100%}.top{align-items:flex-start}.canvasWrap{height:500px}.mag{width:220px;height:220px}}
    </style>

    <div class="app">
      <div class="top">
        <div class="steps">
          <div class="step active" id="step1"><b>1</b> 이미지</div>
          <div class="step" id="step2"><b>2</b> 축 보정</div>
          <div class="step" id="step3"><b>3</b> 값 추출</div>
        </div>
        <div class="upload">
          <input id="file" type="file" accept="image/png,image/jpeg">
          <span id="fileName" class="fileName">PNG/JPG</span>
        </div>
      </div>

      <section class="card workspace">
        <div class="toolbar">
          <div class="btnrow">
            <button id="fit">화면 맞춤</button><button id="one">1:1</button><button id="zin">＋</button><button id="zout">－</button><span id="zoomText" class="zoomText">100%</span>
            <button id="undo">↶ 되돌리기</button><button id="reset">초기화</button>
          </div>
          <div class="btnrow">
            <span class="small">돋보기</span><button class="magbtn" data-mag="5">5×</button><button class="magbtn active" data-mag="8">8×</button><button class="magbtn" data-mag="12">12×</button><button class="magbtn" data-mag="16">16×</button>
          </div>
        </div>
        <div class="canvasWrap empty" id="wrap">
          <div class="stage" id="stage"><canvas id="base"></canvas><canvas id="overlay" class="overlay"></canvas></div>
          <div class="mag" id="mag"><span class="magLabel" id="magLabel">8×</span><canvas id="magCanvas" width="270" height="270"></canvas></div>
        </div>
        <div class="statusbar"><span id="instruction" class="instruction">이미지를 업로드하세요.</span><span id="coord">—</span></div>
        <div class="compactHelp"><b>정밀 조정:</b> 찍은 점을 마우스로 다시 드래그할 수 있습니다 · 마우스 휠로 확대/축소 · 화살표키로 미세 이동</div>
      </section>

      <aside class="side">
        <section class="card">
          <div class="title">축 보정</div>
          <div class="radio-row"><label><input type="radio" name="mode" value="y" checked> Y축만</label><label><input type="radio" name="mode" value="xy"> X·Y축</label></div>
          <label>Y축 기준값 2개</label><div class="cal-grid"><input id="y1v" type="number" value="0" step="any"><input id="y2v" type="number" value="100" step="any"></div>
          <label>Y축 Scale</label><select id="yscale"><option value="linear">Linear</option><option value="log">Log</option></select>
          <div id="xblock" style="display:none"><label>X축 기준값 2개</label><div class="cal-grid"><input id="x1v" type="number" value="0" step="any"><input id="x2v" type="number" value="10" step="any"></div><label>X축 Scale</label><select id="xscale"><option value="linear">Linear</option><option value="log">Log</option></select></div>
          <button id="calStart" class="primary full" disabled>축 보정 시작</button>
          <div id="calMsg" class="hint">이미지를 먼저 선택하세요.</div>
        </section>

        <section class="card">
          <div class="title">값 추출</div>
          <label>오차 종류</label><div class="radio-row" style="display:grid;grid-template-columns:1fr 1fr;gap:6px"><label><input type="radio" name="err" value="sd"> SD</label><label><input type="radio" name="err" value="se" checked> SE</label><label><input type="radio" name="err" value="ci"> 95% CI</label><label><input type="radio" name="err" value="mean"> Mean only</label></div>
          <div class="cal-grid"><div><label>n</label><input id="n" type="number" min="1" value="8"></div><div><label>소수점</label><select id="dec"><option>0</option><option>1</option><option selected>2</option><option>3</option><option>4</option></select></div></div>
          <button id="extractStart" class="primary full" disabled>값 추출 시작</button>
          <div id="extractMsg" class="hint">축 보정 후 사용할 수 있습니다.</div>
          <div class="results" id="results" style="display:none">
            <div class="r" id="xRow" style="display:none"><span>X</span><b id="rX">—</b></div>
            <div class="r"><span id="meanLabel">Mean</span><b class="mean" id="rMean">—</b></div>
            <div class="r" id="errRow"><span id="errLabel">SE</span><b class="err" id="rErr">—</b></div>
            <div class="r" id="sdRow"><span>SD</span><b class="conv" id="rSD">—</b></div>
            <div class="r"><span>n</span><b id="rN">—</b></div>
            <button id="copy" class="full">결과 복사</button><div class="small" id="formula" style="margin-top:6px"></div>
          </div>
        </section>

        <section class="card">
          <div class="title">정밀도</div>
          <div class="sub">선택한 점을 화살표키로 움직이는 간격입니다.</div>
          <div class="precision" style="margin-top:7px"><button class="prec active" data-step="0.25">0.25 px</button><button class="prec" data-step="0.5">0.5 px</button><button class="prec" data-step="1">1 px</button><button class="prec" data-step="2">2 px</button></div>
          <div class="hint">권장: 확대 상태에서 0.25 px. Shift + 화살표는 설정값의 10배 이동합니다.</div>
        </section>
      </aside>
    </div>

    <script>
    (()=>{
      const $=id=>document.getElementById(id), base=$('base'), ov=$('overlay'), bctx=base.getContext('2d'), octx=ov.getContext('2d');
      const wrap=$('wrap'), stage=$('stage'), mag=$('mag'), magCanvas=$('magCanvas'), mctx=magCanvas.getContext('2d');
      let img=null, scale=1, magScale=8, keyStep=.25, action='idle', points=[], cal={}, selected=-1, dragging=false, lastPointer=null;
      const mode=()=>document.querySelector('input[name="mode"]:checked').value;
      const errType=()=>document.querySelector('input[name="err"]:checked').value;
      function setSteps(n){[1,2,3].forEach(i=>{const e=$('step'+i);e.classList.remove('active','done');if(i<n)e.classList.add('done');else if(i===n)e.classList.add('active')})}
      function resizeCanvas(){if(!img)return;base.width=img.naturalWidth;base.height=img.naturalHeight;ov.width=base.width;ov.height=base.height;bctx.clearRect(0,0,base.width,base.height);bctx.drawImage(img,0,0);fit();draw();}
      function setScale(v, anchor=null){if(!img)return;const old=scale;scale=Math.max(.1,Math.min(6,v));applyScale();if(anchor){const ratio=scale/old;wrap.scrollLeft=(wrap.scrollLeft+anchor.x)*ratio-anchor.x;wrap.scrollTop=(wrap.scrollTop+anchor.y)*ratio-anchor.y;}}
      function fit(){if(!img)return;const maxW=Math.max(320,wrap.clientWidth-28),maxH=Math.max(320,wrap.clientHeight-28);setScale(Math.min(1,maxW/base.width,maxH/base.height));}
      function applyScale(){const w=base.width*scale,h=base.height*scale;stage.style.width=w+'px';stage.style.height=h+'px';base.style.width=ov.style.width=w+'px';base.style.height=ov.style.height=h+'px';$('zoomText').textContent=Math.round(scale*100)+'%';}
      function draw(){octx.clearRect(0,0,ov.width,ov.height);const colors=['#16a34a','#2563eb','#ef4444','#f59e0b','#7c3aed','#db2777'];points.forEach((p,i)=>{const r=(i===selected?8:6)/Math.max(scale,.25);octx.beginPath();octx.arc(p.x,p.y,r,0,Math.PI*2);octx.fillStyle=colors[i%colors.length];octx.fill();octx.lineWidth=(i===selected?3:2)/Math.max(scale,.25);octx.strokeStyle=i===selected?'#111827':'#fff';octx.stroke();octx.font=`${12/Math.max(scale,.4)}px sans-serif`;octx.fillStyle=colors[i%colors.length];octx.fillText(p.label,p.x+9/Math.max(scale,.4),p.y-8/Math.max(scale,.4));});}
      function canvasPoint(ev){const r=ov.getBoundingClientRect();return{x:(ev.clientX-r.left)/scale,y:(ev.clientY-r.top)/scale};}
      function hitPoint(p){let best=-1,dist=Infinity;const tol=12/Math.max(scale,.2);points.forEach((q,i)=>{const d=Math.hypot(q.x-p.x,q.y-p.y);if(d<tol&&d<dist){best=i;dist=d}});return best;}
      function updateMagnifier(p){if(!img||!p)return;mag.style.display='block';const s=magCanvas.width,src=s/magScale;const sx=Math.max(0,Math.min(base.width-src,p.x-src/2)),sy=Math.max(0,Math.min(base.height-src,p.y-src/2));mctx.clearRect(0,0,s,s);mctx.imageSmoothingEnabled=false;mctx.drawImage(base,sx,sy,src,src,0,0,s,s);$('magLabel').textContent=magScale+'×';}
      function validLog(v){return Number(v)>0}
      function axisConvert(px,p1,p2,v1,v2,logAxis=false){if(logAxis){const a=Math.log(Number(v1)),b=Math.log(Number(v2));return Math.exp(a+(px-p1)*(b-a)/(p2-p1));}return Number(v1)+(px-p1)*(Number(v2)-Number(v1))/(p2-p1)}
      function yValue(py){return axisConvert(py,cal.y1.y,cal.y2.y,$('y1v').value,$('y2v').value,$('yscale').value==='log')}
      function xValue(px){return axisConvert(px,cal.x1.x,cal.x2.x,$('x1v').value,$('x2v').value,$('xscale').value==='log')}
      function refreshCalRefs(){points.forEach(p=>{if(p.label==='Y1')cal.y1=p;if(p.label==='Y2')cal.y2=p;if(p.label==='X1')cal.x1=p;if(p.label==='X2')cal.x2=p})}
      function startCal(){points=[];cal={};selected=-1;draw();action='cal_y1';$('instruction').textContent='Y1 기준 위치를 클릭하세요.';$('calMsg').className='warnbox';$('calMsg').textContent='Y1 → Y2 순서로 클릭합니다.';setSteps(2);$('extractStart').disabled=true;$('results').style.display='none'}
      function finishCal(){action='idle';selected=points.length-1;$('calMsg').className='okbox';$('calMsg').textContent='✓ 보정 완료. 점은 드래그해 다시 맞출 수 있습니다.';$('extractStart').disabled=false;$('instruction').textContent='축 보정 완료. 값 추출을 시작하세요.';setSteps(3);draw()}
      function startExtract(){points=points.filter(p=>p.kind==='cal');selected=-1;draw();$('results').style.display='none';action='mean';$('instruction').textContent='Mean 위치를 클릭하세요.';$('extractMsg').textContent=errType()==='mean'?'Mean 위치를 클릭합니다.':'Mean → 오차 막대 끝 순서로 클릭합니다.'}
      function recalcFromPoints(){const meanP=[...points].reverse().find(q=>q.label==='Mean'),errP=[...points].reverse().find(q=>q.label==='Error');if(meanP&&(errType()==='mean'||errP))calcResult(meanP,errP)}
      function calcResult(meanP,errP){const d=Number($('dec').value),n=Math.max(1,Number($('n').value)||1),mean=yValue(meanP.y),et=errType();let e=null,sd=null,formula='';if(mode()==='xy'){$('xRow').style.display='grid';$('rX').textContent=xValue(meanP.x).toFixed(d);$('meanLabel').textContent='Y / Mean'}else{$('xRow').style.display='none';$('meanLabel').textContent='Mean'}if(et!=='mean'){const endpoint=yValue(errP.y);e=Math.abs(endpoint-mean);if(et==='se'){sd=e*Math.sqrt(n);formula='SD = SE × √n'}else if(et==='ci'){const se=e/1.96;sd=se*Math.sqrt(n);formula='95% CI를 ±1.96×SE로 간주한 근사값'}$('errRow').style.display='grid';$('sdRow').style.display=(et==='se'||et==='ci')?'grid':'none';$('errLabel').textContent=et==='sd'?'SD (추출값)':et==='se'?'SE (추출값)':'95% CI half-width'}else{$('errRow').style.display='none';$('sdRow').style.display='none'}$('rMean').textContent=mean.toFixed(d);$('rErr').textContent=e==null?'—':e.toFixed(d);$('rSD').textContent=sd==null?'—':sd.toFixed(d);$('rN').textContent=String(n);$('formula').textContent=formula;$('results').style.display='block'}
      function placePoint(p,label,kind){const obj={...p,label,kind};points.push(obj);selected=points.length-1;draw();return obj}
      function handlePlacement(p){if(action==='cal_y1'){cal.y1=placePoint(p,'Y1','cal');action='cal_y2';$('instruction').textContent='Y2 기준 위치를 클릭하세요.'}
        else if(action==='cal_y2'){cal.y2=placePoint(p,'Y2','cal');if(Math.abs(cal.y2.y-cal.y1.y)<1){alert('Y1과 Y2를 서로 다른 높이에 찍어주세요.');points.pop();cal.y2=null;action='cal_y2';draw();return}if(mode()==='xy'){action='cal_x1';$('instruction').textContent='X1 기준 위치를 클릭하세요.'}else finishCal()}
        else if(action==='cal_x1'){cal.x1=placePoint(p,'X1','cal');action='cal_x2';$('instruction').textContent='X2 기준 위치를 클릭하세요.'}
        else if(action==='cal_x2'){cal.x2=placePoint(p,'X2','cal');if(Math.abs(cal.x2.x-cal.x1.x)<1){alert('X1과 X2를 서로 다른 가로 위치에 찍어주세요.');points.pop();cal.x2=null;action='cal_x2';draw();return}finishCal()}
        else if(action==='mean'){placePoint(p,'Mean','data');if(errType()==='mean'){recalcFromPoints();action='idle';$('instruction').textContent='추출 완료. 점을 드래그하면 결과가 즉시 갱신됩니다.'}else{action='error';$('instruction').textContent='오차 막대 끝을 클릭하세요.'}}
        else if(action==='error'){placePoint(p,'Error','data');recalcFromPoints();action='idle';$('instruction').textContent='추출 완료. 점을 드래그하면 결과가 즉시 갱신됩니다.'}}

      $('file').addEventListener('change',e=>{const f=e.target.files&&e.target.files[0];if(!f)return;$('fileName').textContent=f.name;const url=URL.createObjectURL(f),im=new Image();im.onload=()=>{img=im;wrap.classList.remove('empty');resizeCanvas();$('calStart').disabled=false;$('calMsg').className='hint';$('calMsg').textContent='축 보정 시작을 누르세요.';$('instruction').textContent='이미지 준비 완료. 축 보정을 시작하세요.';setSteps(2);URL.revokeObjectURL(url)};im.src=url});
      document.querySelectorAll('input[name="mode"]').forEach(r=>r.addEventListener('change',()=>{$('xblock').style.display=mode()==='xy'?'block':'none';if(img){$('extractStart').disabled=true;$('calMsg').className='warnbox';$('calMsg').textContent='모드를 바꿨습니다. 다시 보정하세요.'}}));
      $('calStart').addEventListener('click',()=>{if($('yscale').value==='log'&&(!validLog($('y1v').value)||!validLog($('y2v').value))){alert('Log Y축 값은 0보다 커야 합니다.');return}if(mode()==='xy'&&$('xscale').value==='log'&&(!validLog($('x1v').value)||!validLog($('x2v').value))){alert('Log X축 값은 0보다 커야 합니다.');return}startCal()});
      $('extractStart').addEventListener('click',startExtract);

      ov.addEventListener('pointermove',e=>{if(!img)return;const p=canvasPoint(e);lastPointer=p;updateMagnifier(p);$('coord').textContent=`pixel x ${p.x.toFixed(2)} · y ${p.y.toFixed(2)}`;if(dragging&&selected>=0){points[selected].x=Math.max(0,Math.min(base.width,p.x));points[selected].y=Math.max(0,Math.min(base.height,p.y));refreshCalRefs();draw();recalcFromPoints()}});
      ov.addEventListener('pointerleave',()=>{if(!dragging)mag.style.display='none'});
      ov.addEventListener('pointerdown',e=>{if(!img)return;const p=canvasPoint(e),hit=hitPoint(p);if(hit>=0){selected=hit;dragging=true;ov.setPointerCapture(e.pointerId);draw();updateMagnifier(points[selected]);e.preventDefault()}else{selected=-1;handlePlacement(p)}});
      ov.addEventListener('pointerup',e=>{if(dragging){dragging=false;try{ov.releasePointerCapture(e.pointerId)}catch(_){ }refreshCalRefs();recalcFromPoints();draw()}});

      wrap.addEventListener('wheel',e=>{if(!img||!e.ctrlKey&&!e.metaKey){if(e.altKey||e.shiftKey){e.preventDefault();const rect=wrap.getBoundingClientRect();setScale(scale*(e.deltaY<0?1.08:.92),{x:e.clientX-rect.left,y:e.clientY-rect.top})}return}e.preventDefault();const rect=wrap.getBoundingClientRect();setScale(scale*(e.deltaY<0?1.12:.89),{x:e.clientX-rect.left,y:e.clientY-rect.top})},{passive:false});
      $('fit').addEventListener('click',fit);$('one').addEventListener('click',()=>setScale(1));$('zin').addEventListener('click',()=>setScale(scale*1.2));$('zout').addEventListener('click',()=>setScale(scale/1.2));
      $('undo').addEventListener('click',()=>{if(!points.length)return;points.pop();selected=points.length-1;refreshCalRefs();draw();$('results').style.display='none';action='idle';$('instruction').textContent='마지막 점을 삭제했습니다.'});
      $('reset').addEventListener('click',()=>{points=[];cal={};selected=-1;draw();$('results').style.display='none';$('extractStart').disabled=true;action='idle';if(img){$('calMsg').className='hint';$('calMsg').textContent='축 보정 시작을 누르세요.';setSteps(2)}});
      document.querySelectorAll('.magbtn').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.magbtn').forEach(x=>x.classList.remove('active'));b.classList.add('active');magScale=Number(b.dataset.mag);$('magLabel').textContent=magScale+'×';if(lastPointer)updateMagnifier(lastPointer)}));
      document.querySelectorAll('.prec').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.prec').forEach(x=>x.classList.remove('active'));b.classList.add('active');keyStep=Number(b.dataset.step)}));
      document.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)||selected<0||!points[selected])return;e.preventDefault();const p=points[selected],d=keyStep*(e.shiftKey?10:1);if(e.key==='ArrowLeft')p.x-=d;if(e.key==='ArrowRight')p.x+=d;if(e.key==='ArrowUp')p.y-=d;if(e.key==='ArrowDown')p.y+=d;p.x=Math.max(0,Math.min(base.width,p.x));p.y=Math.max(0,Math.min(base.height,p.y));refreshCalRefs();draw();updateMagnifier(p);recalcFromPoints()});
      ['n','dec'].forEach(id=>$(id).addEventListener('change',recalcFromPoints));document.querySelectorAll('input[name="err"]').forEach(r=>r.addEventListener('change',()=>{$('results').style.display='none'}));
      $('copy').addEventListener('click',()=>{const lines=[];if($('xRow').style.display!=='none')lines.push(`X\t${$('rX').textContent}`);lines.push(`${$('meanLabel').textContent}\t${$('rMean').textContent}`);if($('errRow').style.display!=='none')lines.push(`${$('errLabel').textContent}\t${$('rErr').textContent}`);if($('sdRow').style.display!=='none')lines.push(`SD\t${$('rSD').textContent}`);lines.push(`n\t${$('rN').textContent}`);const ta=document.createElement('textarea');ta.value=lines.join('\n');document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();$('copy').textContent='✓ 복사됨';setTimeout(()=>$('copy').textContent='결과 복사',1100)});
      new ResizeObserver(()=>{if(img&&scale<1)fit()}).observe(wrap);
    })();
    </script>
    '''
    components.html(html, height=1010, scrolling=False)
