from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def render_figure_digitizer() -> None:
    """Beginner-friendly client-side 2D figure digitizer.

    Everything runs in the browser component: image loading, calibration,
    magnifier, point picking, and SD/SE/CI conversion. No extracted values are
    persisted to the project by design.
    """
    st.markdown(
        """
        <style>
        .digitizer-note {margin-top:-0.25rem;color:#667085;font-size:.92rem}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption("이미지의 2D 그래프에서 필요한 값을 바로 읽는 도구입니다. 추출값은 프로젝트에 저장되지 않습니다.")

    html = r'''
    <style>
      :root{--bg:#f7f9fc;--card:#fff;--line:#d9e0ea;--text:#172033;--muted:#667085;--accent:#2563eb;--ok:#14804a;--warn:#b45309;}
      *{box-sizing:border-box} body{margin:0;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:transparent}
      .app{display:grid;grid-template-columns:250px minmax(0,1fr) 285px;gap:14px;align-items:start}
      .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}
      .title{font-size:16px;font-weight:750;margin:0 0 8px}.sub{font-size:12px;color:var(--muted);line-height:1.5}
      .steps{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:2px}
      .step{border:1px solid var(--line);border-radius:12px;padding:10px 12px;background:#fff;display:flex;gap:9px;align-items:center}
      .badge{width:25px;height:25px;border-radius:999px;background:#eef2f7;color:#4b5563;display:grid;place-items:center;font-weight:750;font-size:13px}
      .step.active .badge,.step.done .badge{background:var(--accent);color:#fff}.step.done{border-color:#bcd0ff}.step strong{font-size:13px}.step span{display:block;color:var(--muted);font-size:11px;margin-top:2px}
      .drop{border:1.5px dashed #9fb1c9;border-radius:12px;padding:17px;text-align:center;background:#fbfdff}.drop input{width:100%;font-size:12px}.hint{margin-top:10px;padding:10px;border-radius:10px;background:#f7f9fc;font-size:12px;color:var(--muted);line-height:1.5}
      label{display:block;font-size:12px;font-weight:650;margin:10px 0 5px}input[type=number],select{width:100%;border:1px solid #cfd8e5;border-radius:8px;padding:8px 9px;background:#fff;color:var(--text);font-size:13px}
      .radio-row{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}.radio-row label{font-weight:500;margin:0;display:flex;gap:5px;align-items:center}
      button{border:1px solid #cfd8e5;background:#fff;border-radius:8px;padding:8px 10px;font-size:12px;cursor:pointer;color:var(--text)}button.primary{background:var(--accent);border-color:var(--accent);color:white;font-weight:700}button:disabled{opacity:.45;cursor:not-allowed}.full{width:100%;margin-top:10px}.btnrow{display:flex;gap:7px;flex-wrap:wrap}.btnrow button.active{background:#eaf1ff;border-color:#8db0ff;color:#174db1}
      .workspace{padding:0;overflow:hidden}.toolbar{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap}.canvasWrap{position:relative;background:#f3f5f8;min-height:500px;display:flex;align-items:center;justify-content:center;overflow:hidden}.canvasWrap.empty:after{content:"① 왼쪽에서 그래프 이미지를 업로드하세요";color:#7b8798;font-size:14px}.stage{position:relative;transform-origin:top left}.stage canvas{display:block;max-width:none}.overlay{position:absolute;left:0;top:0;cursor:crosshair}.statusbar{padding:9px 12px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
      .guide{margin-top:12px;border:1px solid #cfe0ff;background:#f6f9ff;border-radius:12px;padding:12px}.guide strong{font-size:13px;color:#174db1}.guide ol{margin:8px 0 0 19px;padding:0;font-size:12px;line-height:1.65}
      .cal-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.pointline{display:flex;gap:7px;align-items:center;font-size:12px;margin:7px 0}.dot{width:9px;height:9px;border-radius:50%}.green{background:#22a447}.blue{background:#2563eb}.red{background:#ef4444}.orange{background:#f59e0b}
      .okbox{margin-top:10px;border:1px solid #b9dfc8;background:#f2fbf5;border-radius:9px;padding:9px;font-size:12px;color:#17623a}.warnbox{margin-top:10px;border:1px solid #f1d39b;background:#fffbeb;border-radius:9px;padding:9px;font-size:12px;color:#8a5509}
      .results{margin-top:10px;border:1px solid #f1d39b;background:#fffaf0;border-radius:11px;padding:11px}.results h4{margin:0 0 8px;font-size:13px}.r{display:grid;grid-template-columns:1fr auto;gap:8px;padding:7px 0;border-top:1px solid #eee4cf;font-size:13px}.r:first-of-type{border-top:0}.r b{font-size:15px}.r .mean{color:#1d4ed8}.r .err{color:#dc2626}.r .conv{color:#16803a}.small{font-size:11px;color:var(--muted);line-height:1.45}
      .mag{position:absolute;right:12px;bottom:12px;width:150px;height:150px;border:2px solid #fff;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.2);background:#fff;display:none;overflow:hidden}.mag canvas{width:100%;height:100%}.mag:before,.mag:after{content:"";position:absolute;background:#ef4444;z-index:2;pointer-events:none}.mag:before{width:1px;height:100%;left:50%;top:0}.mag:after{height:1px;width:100%;top:50%;left:0}
      @media(max-width:1050px){.app{grid-template-columns:220px 1fr}.right{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;gap:12px}.steps{grid-column:1/-1}}
      @media(max-width:760px){.app{display:block}.card,.workspace,.right{margin-bottom:12px}.right{display:block}.steps{display:block}.step{margin-bottom:7px}.canvasWrap{min-height:400px}}
    </style>

    <div class="app">
      <div class="steps">
        <div class="step active" id="step1"><div class="badge">1</div><div><strong>이미지 업로드</strong><span>그래프 파일을 선택하세요</span></div></div>
        <div class="step" id="step2"><div class="badge">2</div><div><strong>축 보정</strong><span>기준점을 찍고 실제 값을 입력하세요</span></div></div>
        <div class="step" id="step3"><div class="badge">3</div><div><strong>값 추출</strong><span>평균과 오차 막대를 클릭하세요</span></div></div>
      </div>

      <section class="card">
        <div class="title">1. 이미지 업로드</div>
        <div class="drop"><div style="font-size:26px">▧</div><div style="font-weight:700;margin:5px 0">그래프 이미지 선택</div><div class="sub">PNG · JPG · JPEG</div><input id="file" type="file" accept="image/png,image/jpeg" style="margin-top:12px"></div>
        <div class="hint">TIP · 논문 PDF의 Figure를 캡처한 이미지도 사용할 수 있습니다. 축 숫자가 선명할수록 정확하게 보정하기 쉽습니다.</div>
      </section>

      <section class="card workspace">
        <div class="toolbar">
          <div class="btnrow"><button id="fit">화면 맞춤</button><button id="zin">＋ 확대</button><button id="zout">－ 축소</button><button id="undo">↶ 마지막 점</button><button id="reset">전체 초기화</button></div>
          <div class="btnrow"><span class="small" style="align-self:center">돋보기</span><button class="magbtn" data-mag="3">3×</button><button class="magbtn active" data-mag="5">5×</button><button class="magbtn" data-mag="8">8×</button><button class="magbtn" data-mag="10">10×</button></div>
        </div>
        <div class="canvasWrap empty" id="wrap">
          <div class="stage" id="stage"><canvas id="base"></canvas><canvas id="overlay" class="overlay"></canvas></div>
          <div class="mag" id="mag"><canvas id="magCanvas" width="150" height="150"></canvas></div>
        </div>
        <div class="statusbar"><span id="instruction">이미지를 업로드하면 순서대로 안내합니다.</span><span>클릭 후 화살표키: 1px 미세 이동 · Shift+화살표: 10px</span></div>
        <div class="guide"><strong>처음 사용한다면 이것만 따라하세요</strong><ol><li>이미지를 업로드합니다.</li><li>오른쪽의 <b>축 보정 시작</b>을 누르고 화면 안내대로 기준점을 클릭합니다.</li><li><b>값 추출 시작</b>을 누른 뒤 평균 → 오차 막대 끝 순서로 클릭합니다.</li></ol></div>
      </section>

      <aside class="right">
        <section class="card">
          <div class="title">2. 축 보정</div>
          <div class="radio-row"><label><input type="radio" name="mode" value="y" checked> Y축만 (Bar graph)</label><label><input type="radio" name="mode" value="xy"> X·Y축 (2D plot)</label></div>
          <label>Y축 기준값</label><div class="cal-grid"><input id="y1v" type="number" value="0" step="any"><input id="y2v" type="number" value="100" step="any"></div>
          <div class="small">왼쪽 값의 위치(Y1) → 오른쪽 값의 위치(Y2) 순으로 클릭</div>
          <label>Y축 Scale</label><select id="yscale"><option value="linear">Linear</option><option value="log">Log</option></select>
          <div id="xblock" style="display:none"><label>X축 기준값</label><div class="cal-grid"><input id="x1v" type="number" value="0" step="any"><input id="x2v" type="number" value="10" step="any"></div><div class="small">X1 → X2 순으로 클릭</div><label>X축 Scale</label><select id="xscale"><option value="linear">Linear</option><option value="log">Log</option></select></div>
          <button id="calStart" class="primary full" disabled>축 보정 시작</button>
          <div id="calMsg" class="warnbox">이미지를 먼저 업로드하세요.</div>
        </section>

        <section class="card" style="margin-top:12px">
          <div class="title">3. 값 추출</div>
          <label>오차 막대 종류</label><div class="radio-row" style="display:grid;grid-template-columns:1fr 1fr;gap:7px"><label><input type="radio" name="err" value="sd"> SD</label><label><input type="radio" name="err" value="se" checked> SE</label><label><input type="radio" name="err" value="ci"> 95% CI</label><label><input type="radio" name="err" value="mean"> Mean only</label></div>
          <label>표본 크기 (n)</label><input id="n" type="number" min="1" value="8">
          <label>소수점 자리</label><select id="dec"><option>0</option><option>1</option><option selected>2</option><option>3</option><option>4</option></select>
          <button id="extractStart" class="primary full" disabled>값 추출 시작</button>
          <div id="extractMsg" class="hint">축 보정을 완료하면 사용할 수 있습니다.</div>
          <div class="results" id="results" style="display:none" aria-live="polite"><h4>추출 결과</h4><div class="r" id="xRow" style="display:none"><span>X</span><b id="rX">—</b></div><div class="r"><span id="meanLabel">Mean</span><b class="mean" id="rMean">—</b></div><div class="r" id="errRow"><span id="errLabel">SE (추출값)</span><b class="err" id="rErr">—</b></div><div class="r" id="sdRow"><span>SD (계산값)</span><b class="conv" id="rSD">—</b></div><div class="r"><span>n</span><b id="rN">—</b></div><button id="copy" class="full">결과 복사</button><div class="small" id="formula" style="margin-top:8px"></div></div>
        </section>
      </aside>
    </div>

    <script>
    (()=>{
      const $=id=>document.getElementById(id), base=$('base'), ov=$('overlay'), bctx=base.getContext('2d'), octx=ov.getContext('2d');
      const wrap=$('wrap'), stage=$('stage'), mag=$('mag'), mctx=$('magCanvas').getContext('2d');
      let img=null, scale=1, magScale=5, action='idle', points=[], cal={}, lastMouse=null;
      const mode=()=>document.querySelector('input[name="mode"]:checked').value;
      const errType=()=>document.querySelector('input[name="err"]:checked').value;
      const setSteps=n=>{[1,2,3].forEach(i=>{const e=$('step'+i);e.classList.remove('active','done');if(i<n)e.classList.add('done');else if(i===n)e.classList.add('active')})};
      function resizeCanvas(){if(!img)return;base.width=img.naturalWidth;base.height=img.naturalHeight;ov.width=base.width;ov.height=base.height;bctx.clearRect(0,0,base.width,base.height);bctx.drawImage(img,0,0);draw();fit();}
      function fit(){if(!img)return;const maxW=Math.max(320,wrap.clientWidth-24), maxH=610;scale=Math.min(1,maxW/base.width,maxH/base.height);applyScale();}
      function applyScale(){stage.style.width=(base.width*scale)+'px';stage.style.height=(base.height*scale)+'px';base.style.width=ov.style.width=(base.width*scale)+'px';base.style.height=ov.style.height=(base.height*scale)+'px';}
      function draw(){octx.clearRect(0,0,ov.width,ov.height);const colors=['#16a34a','#2563eb','#ef4444','#f59e0b','#7c3aed','#db2777'];points.forEach((p,i)=>{octx.beginPath();octx.arc(p.x,p.y,6/scale,0,Math.PI*2);octx.fillStyle=colors[i%colors.length];octx.fill();octx.lineWidth=2/scale;octx.strokeStyle='#fff';octx.stroke();octx.font=`${12/scale}px sans-serif`;octx.fillStyle=colors[i%colors.length];octx.fillText(p.label,p.x+9/scale,p.y-8/scale);});}
      function canvasPoint(ev){const r=ov.getBoundingClientRect();return {x:(ev.clientX-r.left)/scale,y:(ev.clientY-r.top)/scale};}
      function updateMagnifier(p){if(!img)return;mag.style.display='block';const s=150, src=s/magScale;const sx=Math.max(0,Math.min(base.width-src,p.x-src/2)), sy=Math.max(0,Math.min(base.height-src,p.y-src/2));mctx.clearRect(0,0,s,s);mctx.imageSmoothingEnabled=false;mctx.drawImage(base,sx,sy,src,src,0,0,s,s);}
      function validLog(v){return Number(v)>0}
      function axisConvert(px,p1,p2,v1,v2,logAxis=false){if(logAxis){const a=Math.log(Number(v1)),b=Math.log(Number(v2));return Math.exp(a+(px-p1)*(b-a)/(p2-p1));}return Number(v1)+(px-p1)*(Number(v2)-Number(v1))/(p2-p1);}
      function yValue(py){return axisConvert(py,cal.y1.y,cal.y2.y,$('y1v').value,$('y2v').value,$('yscale').value==='log');}
      function xValue(px){return axisConvert(px,cal.x1.x,cal.x2.x,$('x1v').value,$('x2v').value,$('xscale').value==='log');}
      function calReady(){return !!(cal.y1&&cal.y2&&(mode()==='y'||(cal.x1&&cal.x2)));}
      function startCal(){points=[];cal={};draw();action='cal_y1';$('instruction').textContent='Y1: 첫 번째 Y축 기준값 위치를 클릭하세요.';$('calMsg').className='warnbox';$('calMsg').textContent='Y1 위치를 클릭하세요.';setSteps(2);$('extractStart').disabled=true;$('results').style.display='none';}
      function finishCal(){action='idle';$('calMsg').className='okbox';$('calMsg').textContent='✓ 축 보정 완료. 이제 값을 추출할 수 있습니다.';$('extractStart').disabled=false;$('instruction').textContent='축 보정 완료. 오른쪽에서 값 추출 시작을 누르세요.';setSteps(3);}
      function startExtract(){points=points.filter(p=>p.kind==='cal');draw();$('results').style.display='none';const et=errType();action='mean';$('instruction').textContent='평균(Mean): 막대의 높이 또는 데이터 점 중심을 클릭하세요.';$('extractMsg').textContent=et==='mean'?'평균 위치를 클릭하세요.':'① 평균 위치 → ② 오차 막대 끝 순서로 클릭하세요.';}
      function calcResult(meanP,errP){const d=Number($('dec').value), n=Math.max(1,Number($('n').value)||1), mean=yValue(meanP.y), et=errType();let e=null,sd=null,formula='';if(mode()==='xy'){$('xRow').style.display='grid';$('rX').textContent=xValue(meanP.x).toFixed(d);$('meanLabel').textContent='Y / Mean';}else{$('xRow').style.display='none';$('meanLabel').textContent='Mean';}if(et!=='mean'){const endpoint=yValue(errP.y);e=Math.abs(endpoint-mean);if(et==='se'){sd=e*Math.sqrt(n);formula=`SD = SE × √n`;}else if(et==='ci'){const se=e/1.96;sd=se*Math.sqrt(n);formula='95% CI를 ±1.96×SE로 간주한 근사 SD입니다.';}$('errRow').style.display='grid';$('sdRow').style.display=(et==='se'||et==='ci')?'grid':'none';$('errLabel').textContent=et==='sd'?'SD (추출값)':et==='se'?'SE (추출값)':'95% CI half-width';$('sdRow').querySelector('span').textContent=et==='ci'?'SD (근사 계산값)':'SD (계산값)';}else{$('errRow').style.display='none';$('sdRow').style.display='none';}
        $('rMean').textContent=mean.toFixed(d);$('rErr').textContent=e==null?'—':e.toFixed(d);$('rSD').textContent=sd==null?'—':sd.toFixed(d);$('rN').textContent=String(n);$('formula').textContent=formula;$('results').style.display='block';$('extractMsg').textContent='완료. 다른 값을 추출하려면 “값 추출 시작”을 다시 누르세요.';}
      $('file').addEventListener('change',e=>{const f=e.target.files&&e.target.files[0];if(!f)return;const url=URL.createObjectURL(f);const im=new Image();im.onload=()=>{img=im;wrap.classList.remove('empty');resizeCanvas();$('calStart').disabled=false;$('calMsg').className='hint';$('calMsg').textContent='축 보정 시작을 누르세요.';$('instruction').textContent='이미지 업로드 완료. 오른쪽에서 축 보정을 시작하세요.';setSteps(2);URL.revokeObjectURL(url)};im.src=url;});
      document.querySelectorAll('input[name="mode"]').forEach(r=>r.addEventListener('change',()=>{$('xblock').style.display=mode()==='xy'?'block':'none';if(img){$('extractStart').disabled=true;$('calMsg').className='warnbox';$('calMsg').textContent='모드를 변경했습니다. 축 보정을 다시 해주세요.';}}));
      $('calStart').addEventListener('click',()=>{if($('yscale').value==='log'&&(!validLog($('y1v').value)||!validLog($('y2v').value))){alert('Log Y축 기준값은 0보다 커야 합니다.');return}if(mode()==='xy'&&$('xscale').value==='log'&&(!validLog($('x1v').value)||!validLog($('x2v').value))){alert('Log X축 기준값은 0보다 커야 합니다.');return}startCal();});
      $('extractStart').addEventListener('click',startExtract);
      ov.addEventListener('mousemove',e=>{if(!img)return;lastMouse=canvasPoint(e);updateMagnifier(lastMouse)});ov.addEventListener('mouseleave',()=>mag.style.display='none');
      ov.addEventListener('click',e=>{if(!img)return;const p=canvasPoint(e);if(action==='cal_y1'){cal.y1=p;points.push({...p,label:'Y1',kind:'cal'});action='cal_y2';$('instruction').textContent='Y2: 두 번째 Y축 기준값 위치를 클릭하세요.';$('calMsg').textContent='Y2 위치를 클릭하세요.';}
        else if(action==='cal_y2'){cal.y2=p;points.push({...p,label:'Y2',kind:'cal'});if(Math.abs(cal.y2.y-cal.y1.y)<2){alert('Y1과 Y2는 서로 다른 높이에 찍어주세요.');points.pop();cal.y2=null;return}if(mode()==='xy'){action='cal_x1';$('instruction').textContent='X1: 첫 번째 X축 기준값 위치를 클릭하세요.';$('calMsg').textContent='X1 위치를 클릭하세요.';}else finishCal();}
        else if(action==='cal_x1'){cal.x1=p;points.push({...p,label:'X1',kind:'cal'});action='cal_x2';$('instruction').textContent='X2: 두 번째 X축 기준값 위치를 클릭하세요.';$('calMsg').textContent='X2 위치를 클릭하세요.';}
        else if(action==='cal_x2'){cal.x2=p;points.push({...p,label:'X2',kind:'cal'});if(Math.abs(cal.x2.x-cal.x1.x)<2){alert('X1과 X2는 서로 다른 가로 위치에 찍어주세요.');points.pop();cal.x2=null;return}finishCal();}
        else if(action==='mean'){points.push({...p,label:'Mean',kind:'data'});if(errType()==='mean'){calcResult(p,null);action='idle';}else{action='error';$('instruction').textContent='오차 막대 끝(Upper 또는 Lower)을 클릭하세요.';}}
        else if(action==='error'){points.push({...p,label:'Error',kind:'data'});const mp=points.slice().reverse().find(q=>q.label==='Mean');calcResult(mp,p);action='idle';$('instruction').textContent='추출 완료.';}draw();});
      $('fit').addEventListener('click',fit);$('zin').addEventListener('click',()=>{scale=Math.min(4,scale*1.2);applyScale()});$('zout').addEventListener('click',()=>{scale=Math.max(.1,scale/1.2);applyScale()});
      $('undo').addEventListener('click',()=>{if(!points.length)return;points.pop();draw();$('results').style.display='none';$('instruction').textContent='마지막 점을 삭제했습니다. 필요한 단계를 다시 시작해 주세요.';action='idle';});
      $('reset').addEventListener('click',()=>{points=[];cal={};draw();$('results').style.display='none';$('extractStart').disabled=true;action='idle';if(img){$('calMsg').className='hint';$('calMsg').textContent='축 보정 시작을 누르세요.';setSteps(2)}});
      document.querySelectorAll('.magbtn').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.magbtn').forEach(x=>x.classList.remove('active'));b.classList.add('active');magScale=Number(b.dataset.mag)}));
      document.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)||!points.length)return;e.preventDefault();const p=points[points.length-1],d=e.shiftKey?10:1;if(e.key==='ArrowLeft')p.x-=d;if(e.key==='ArrowRight')p.x+=d;if(e.key==='ArrowUp')p.y-=d;if(e.key==='ArrowDown')p.y+=d;draw();if(p.label==='Y1')cal.y1=p;if(p.label==='Y2')cal.y2=p;if(p.label==='X1')cal.x1=p;if(p.label==='X2')cal.x2=p;});
      $('copy').addEventListener('click',()=>{const lines=[];if($('xRow').style.display!=='none')lines.push(`X\t${$('rX').textContent}`);lines.push(`${$('meanLabel').textContent}\t${$('rMean').textContent}`);if($('errRow').style.display!=='none')lines.push(`${$('errLabel').textContent}\t${$('rErr').textContent}`);if($('sdRow').style.display!=='none')lines.push(`SD\t${$('rSD').textContent}`);lines.push(`n\t${$('rN').textContent}`);const ta=document.createElement('textarea');ta.value=lines.join('\n');document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();$('copy').textContent='✓ 복사됨';setTimeout(()=>$('copy').textContent='결과 복사',1200);});
      new ResizeObserver(()=>{if(img&&scale<=1)fit()}).observe(wrap);
    })();
    </script>
    '''
    components.html(html, height=1050, scrolling=True)
