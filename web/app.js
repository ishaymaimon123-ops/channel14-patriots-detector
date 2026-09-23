const $ = (id) => document.getElementById(id);
const state = {file: null, jobId: null, pollTimer: null};
const fmt = (s) => {s = Math.max(0, Math.floor(Number(s)||0)); return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`};
const percent = (n) => `${Math.round((Number(n)||0)*100)}%`;
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function setProgress(stage, value) {
  $('progressPanel').hidden = false;
  $('progressStage').textContent = stage;
  $('progressPercent').textContent = percent(value);
  $('progressFill').style.width = percent(value);
}
function setError(message) {$('errorMessage').textContent=message; $('errorMessage').hidden=false;}
function choose(file) {
  if (!file) return;
  state.file=file;
  $('fileRow').hidden=false;
  $('fileName').textContent=file.name;
  $('fileSize').textContent=`${(file.size/1024/1024).toFixed(1)} MB`;
  $('analyseButton').disabled=false;
  $('errorMessage').hidden=true;
  $('resultPanel').hidden=true;
}

async function loadReference() {
  try {
    const health = await fetch('/api/health').then(r=>r.json());
    $('serverStatus').classList.toggle('ready',health.ok && health.ffmpeg);
    $('serverStatus').innerHTML=`<span class="status-dot"></span> ${health.ffmpeg?'המערכת מוכנה':'כלי הווידאו לא זמין'}`;
    const r = await fetch('/api/reference-test').then(r=>r.json());
    $('referenceClass').textContent=r.classification==='COMMERCIAL'?'הפסקת פרסומות':'תוכנית';
    $('referenceConfidence').textContent=percent(r.confidence);
    $('leftScore').textContent=Number(r.upper_left_box_score).toFixed(2);
    $('rightScore').textContent=Number(r.upper_right_symbol_score).toFixed(2);
    $('patriotsScore').textContent=Number(r.patriots_text_score).toFixed(2);
    $('timerValue').textContent=r.detected_timer||'—';
    $('returnScore').textContent=Number(r.return_soon_text_score).toFixed(2);
  } catch(e) {
    $('serverStatus').textContent='השרת אינו זמין';
    $('referenceClass').textContent='לא זמין';
  }
}

function upload() {
  if (!state.file) return;
  if (state.pollTimer) clearTimeout(state.pollTimer);
  $('analyseButton').disabled=true;
  $('analyseLinkButton').disabled=true;
  $('demoButton').disabled=true;
  $('errorMessage').hidden=true;
  setProgress('מעלה את הסרטון',0);
  const xhr = new XMLHttpRequest();
  xhr.open('POST','/api/jobs');
  xhr.setRequestHeader('Content-Type','application/octet-stream');
  xhr.setRequestHeader('X-Filename',encodeURIComponent(state.file.name));
  xhr.upload.onprogress = e => {if(e.lengthComputable) setProgress('מעלה את הסרטון',.05+.25*e.loaded/e.total)};
  xhr.onerror = () => {setError('ההעלאה נכשלה. בדוק שהשרת פועל ונסה שוב.');$('analyseButton').disabled=false;$('analyseLinkButton').disabled=false;$('demoButton').disabled=false};
  xhr.onload = () => {
    let data={}; try{data=JSON.parse(xhr.responseText)}catch{}
    if(xhr.status!==202){setError(data.error||'ההעלאה נכשלה');$('analyseButton').disabled=false;$('analyseLinkButton').disabled=false;$('demoButton').disabled=false;return}
    state.jobId=data.id;
    localStorage.setItem('patriotsJobId', data.id);
    setProgress('מתחיל ניתוח',.01);
    poll();
  };
  xhr.send(state.file);
}

async function runLink() {
  const url=$('videoUrl').value.trim();
  if(!/^https?:\/\//i.test(url)){setError('יש להדביק קישור HTTP או HTTPS ישיר לווידאו או ל־HLS');return}
  if (state.pollTimer) clearTimeout(state.pollTimer);
  $('analyseLinkButton').disabled=true;
  $('analyseButton').disabled=true;
  $('demoButton').disabled=true;
  $('errorMessage').hidden=true;
  $('resultPanel').hidden=true;
  setProgress('בודק את הקישור',.01);
  try {
    const response=await fetch('/api/url-jobs',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url,mode:$('linkMode').value,force:$('forceLinkScan').checked})});
    const data=await response.json();
    if(!response.ok) throw Error(data.error||'לא ניתן לקבל את הקישור');
    state.jobId=data.id;
    localStorage.setItem('patriotsJobId',data.id);
    if(data.reused) setProgress('טוען תוצאה שמורה',1);
    $('forceLinkScan').checked=false;
    poll();
  } catch(e) {
    setError(e.message);
    $('analyseLinkButton').disabled=false;
    $('analyseButton').disabled=!state.file;
    $('demoButton').disabled=false;
  }
}

async function runDemo() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  $('demoButton').disabled=true;
  $('analyseButton').disabled=true;
  $('analyseLinkButton').disabled=true;
  $('errorMessage').hidden=true;
  $('resultPanel').hidden=true;
  setProgress('מכין סרטון הדגמה',.01);
  try {
    const response=await fetch('/api/demo',{method:'POST'});
    const data=await response.json();
    if(!response.ok) throw Error(data.error||'סרטון ההדגמה לא זמין');
    state.jobId=data.id;
    localStorage.setItem('patriotsJobId',data.id);
    poll();
  } catch(e) {
    setError(e.message);
    $('demoButton').disabled=false;
    $('analyseButton').disabled=!state.file;
    $('analyseLinkButton').disabled=false;
  }
}

async function poll(restoring=false) {
  if (!state.jobId) return;
  try {
    const response=await fetch(`/api/jobs/${state.jobId}`);
    if(!response.ok) throw Error('לא ניתן לקרוא את מצב הניתוח');
    const job=await response.json();
    setProgress(job.stage||'מנתח',job.progress||0);
    if(job.status==='done'){
      renderResult(job.result);
      $('analyseButton').disabled=!state.file;
      $('demoButton').disabled=false;
      $('analyseLinkButton').disabled=false;
      $('analyseButton').textContent=state.file?'נתח שוב את הסרטון':'התחל ניתוח';
      return;
    }
    if(job.status==='error'){
      setError(job.error||'הניתוח נכשל');
      $('analyseButton').disabled=false;
      $('demoButton').disabled=false;
      $('analyseLinkButton').disabled=false;
      return;
    }
    state.pollTimer=setTimeout(poll,1200);
  } catch(e) {
    localStorage.removeItem('patriotsJobId');
    if (restoring) $('progressPanel').hidden=true;
    else setError(e.message);
    $('analyseButton').disabled=!state.file;
    $('demoButton').disabled=false;
    $('analyseLinkButton').disabled=false;
  }
}

function renderResult(result) {
  const breaks=result.breaks||[];
  $('adPlayer').pause();
  $('adPlayer').removeAttribute('src');
  $('adPlayer').load();
  const ads=breaks.reduce((sum,b)=>sum+(b.ads||[]).length,0);
  $('durationStat').textContent=fmt(result.duration);
  $('breakStat').textContent=breaks.length;
  $('adStat').textContent=ads;
  $('frameStat').textContent=result.sample_count;
  if(result.mode){
    $('sourceNote').hidden=false;
    $('sourceNote').textContent=result.mode==='full' ? `נותח כל הקישור (${fmt(result.duration)}).` :
      `נותח קטע באורך ${fmt(result.duration)} מתחילת הקישור (${result.source_duration?`האורך הכולל ${fmt(result.source_duration)}`:'שידור חי'}). אפשר לבחור ״כל הקישור״ לניתוח מלא.`;
  } else $('sourceNote').hidden=true;
  $('allAdsDownload').hidden=!result.all_ads_url;
  $('viewerPanel').hidden=!result.all_ads_url;
  if(result.all_ads_url) {
    $('allAdsDownload').href=result.all_ads_url+'?download=1';
    $('adPlayer').src=result.all_ads_url;
  }
  $('timelineEnd').textContent=fmt(result.duration);
  $('timeline').innerHTML=breaks.map(b=>{
    const left=Math.max(0,100*b.start/result.duration);
    const width=Math.max(.4,100*(b.end-b.start)/result.duration);
    return `<div class="segment" style="right:${left}%;width:${width}%" title="${fmt(b.start)}–${fmt(b.end)}"></div>`;
  }).join('');
  $('resultPanel').hidden=false;
  $('resultPanel').scrollIntoView({behavior:'smooth',block:'start'});
}

$('dropzone').addEventListener('click',()=>$('fileInput').click());
$('dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('fileInput').click()}});
$('fileInput').addEventListener('change',e=>choose(e.target.files[0]));
for(const event of ['dragenter','dragover']) $('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('dragging')});
for(const event of ['dragleave','drop']) $('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('dragging')});
$('dropzone').addEventListener('drop',e=>choose(e.dataTransfer.files[0]));
$('analyseButton').addEventListener('click',upload);
$('analyseLinkButton').addEventListener('click',runLink);
$('videoUrl').addEventListener('keydown',e=>{if(e.key==='Enter')runLink()});
$('demoButton').addEventListener('click',runDemo);
loadReference();
const requestedJob = new URLSearchParams(location.search).get('job');
if(requestedJob && /^[0-9a-f]{32}$/.test(requestedJob)) localStorage.setItem('patriotsJobId',requestedJob);
const savedJob = localStorage.getItem('patriotsJobId');
if (savedJob && /^[0-9a-f]{32}$/.test(savedJob)) {state.jobId=savedJob; poll(true)}
