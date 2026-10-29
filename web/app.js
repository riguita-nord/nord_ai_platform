const $=s=>document.querySelector(s);let current=null,busy=false,models=[],selected="";
async function api(path,data){const r=await fetch("/api/"+path,{method:data===undefined?"GET":"POST",headers:{"Content-Type":"application/json"},credentials:"same-origin",...(data===undefined?{}:{body:JSON.stringify(data)})});const j=await r.json();if(!r.ok)throw Error(j.error||"Erro de ligação");return j}
function showApp(ok){$("#login").classList.toggle("hidden",ok);$("#platform").classList.toggle("hidden",!ok)}
function textNode(role,content){const el=document.createElement("div");el.className="msg "+role;el.textContent=content;$("#messages").append(el);$("#chat").scrollTop=$("#chat").scrollHeight;return el}
function toggleWelcome(){ $("#welcome").classList.toggle("hidden",$("#messages").children.length>0)}
async function loadChats(){const j=await api("conversations");const root=$("#conversations");root.replaceChildren();for(const c of j.conversations){const b=document.createElement("button");b.className="conversation"+(c.id===current?" active":"");b.textContent="▤  "+c.title;b.onclick=()=>openChat(c.id);root.append(b)}}
async function openChat(id){current=id;$("#messages").replaceChildren();const j=await api("conversations/"+encodeURIComponent(id));j.messages.forEach(m=>textNode(m.role,m.content));toggleWelcome();loadChats();$(".sidebar").classList.remove("open")}
function newChat(){current=null;$("#messages").replaceChildren();toggleWelcome();loadChats();$(".sidebar").classList.remove("open")}
async function loadModels(){const j=await api("models");models=j.models;selected=models.includes(selected)?selected:(models.includes(j.preferred)?j.preferred:(models[0]||""));$("#modelSelect").replaceChildren();for(const m of models){const opt=document.createElement("option");opt.value=m;opt.textContent=m;$("#modelSelect").append(opt)}$("#modelSelect").value=selected;$("#modelCount").textContent=models.length+" instalado(s)";const box=$("#installedModels");box.replaceChildren();for(const m of models){const row=document.createElement("div");row.className="model-row";const label=document.createElement("strong");label.textContent="✳  "+m;const del=document.createElement("button");del.textContent="Remover";del.onclick=async()=>{if(!confirm("Remover "+m+"?"))return;try{await api("delete-model",{name:m});await loadModels()}catch(e){alert(e.message)}};row.append(label,del);box.append(row)}if(!models.length)box.textContent="Ainda não há modelos instalados."}
async function watchJobs(){try{const j=await api("jobs");const jobs=Object.values(j.jobs);const latest=jobs.at(-1);if(latest){$("#jobStatus").textContent=latest.name+" — "+latest.status+(latest.percent===null||latest.percent===undefined?"":" ("+latest.percent+"%)");if(!latest.done){setTimeout(watchJobs,1300)}else{await loadModels()}}}catch(e){$("#jobStatus").textContent=e.message}}
async function send(){
 if(busy)return;
 const prompt=$("#prompt").value.trim();if(!prompt)return;
 if(!selected)return alert("Instala um modelo primeiro.");
 busy=true;$("#send").disabled=true;$("#prompt").value="";
 textNode("user",prompt);const el=textNode("assistant","");
 toggleWelcome();
 const indicator=$("#activityStatus");
 indicator.classList.remove("hidden");
 indicator.textContent="◌ A preparar resposta · Sem pesquisa na Internet";
 let full="",completed=false,sawDone=false,searched=false;
 try{
  const res=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},credentials:"same-origin",body:JSON.stringify({model:selected,prompt,conversation_id:current})});
  if(!res.ok){const e=await res.json();throw Error(e.error||"Falha no chat")}
  const reader=res.body.getReader(),decoder=new TextDecoder();let buf="";
  const handle=line=>{
   if(!line.trim())return;
   const event=JSON.parse(line);
   if(event.conversation_id)current=event.conversation_id;
   if(event.error)throw Error(event.error);
   if(event.search_performed)searched=true;
   if(event.status)indicator.textContent="◌ "+event.status+(searched?" · Pesquisa Web":" · Sem pesquisa na Internet");
   if(event.sources?.length){indicator.textContent="◌ Fontes encontradas: "+event.sources.map(s=>s.title).join(", ");}
   if(event.text){full+=event.text;el.textContent=full;$("#chat").scrollTop=$("#chat").scrollHeight}
   if(event.done){sawDone=true;completed=!!event.complete}
  };
  while(true){
   const {done,value}=await reader.read();if(done)break;
   buf+=decoder.decode(value,{stream:true});let ix;
   while((ix=buf.indexOf("\n"))>=0){const line=buf.slice(0,ix);buf=buf.slice(ix+1);handle(line)}
  }
  buf+=decoder.decode();if(buf.trim())handle(buf);
  if(!full)el.textContent="O modelo não devolveu texto.";
  if(!sawDone||!completed){
   el.textContent+=(full?"\n\n":"")+"[Resposta interrompida. Podes pedir para continuar.]";
   indicator.textContent="⚠ Resposta incompleta"+(searched?" · Pesquisa efetuada":" · Sem pesquisa");
  }else indicator.textContent="✓ Resposta concluída"+(searched?" · Pesquisa efetuada":" · Sem pesquisa");
  await loadChats();
 }catch(e){
  el.textContent=(full?full+"\n\n":"")+"Erro: "+e.message;
  indicator.textContent="⚠ Erro na geração";
 }finally{busy=false;$("#send").disabled=false;$("#prompt").focus()}
}
$("#loginForm").onsubmit=async e=>{e.preventDefault();try{await api("login",{password:$("#password").value});$("#password").value="";$("#loginError").textContent="";showApp(true);await Promise.all([loadChats(),loadModels()])}catch(e){$("#loginError").textContent=e.message}};
$("#logout").onclick=async()=>{await api("logout",{});showApp(false)};
$("#newChat").onclick=newChat;$("#menuBtn").onclick=()=>$(".sidebar").classList.toggle("open");
$("#modelsBtn").onclick=()=>{$("#modal").classList.remove("hidden");loadModels();watchJobs()};
$("#closeModal").onclick=()=>$("#modal").classList.add("hidden");
$("#modal").onclick=e=>{if(e.target===$("#modal"))$("#modal").classList.add("hidden")};
$("#installModel").onclick=async()=>{try{const j=await api("pull",{name:$("#addModel").value});$("#jobStatus").textContent="Download iniciado: "+j.job;watchJobs()}catch(e){$("#jobStatus").textContent=e.message}};
$("#modelSelect").onchange=e=>selected=e.target.value;
$("#send").onclick=send;$("#prompt").onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}};
document.querySelectorAll("[data-prompt]").forEach(b=>b.onclick=()=>{$("#prompt").value=b.dataset.prompt;send()});
api("me").then(j=>{showApp(j.authenticated);if(j.authenticated)Promise.all([loadChats(),loadModels()])}).catch(()=>showApp(false));

async function loadLearning(){
 const data=await api("learning");const box=$("#learnItems");box.replaceChildren();
 $("#learnCount").textContent=data.items.length+" apresentados";
 for(const item of data.items){
   const row=document.createElement("article");row.className="learn-item";
   const header=document.createElement("div");header.className="model-title";
   const title=document.createElement("strong");title.textContent=item.title;
   const del=document.createElement("button");del.textContent="Remover";del.className="ghost";
   del.onclick=async()=>{if(!confirm("Remover este conhecimento?"))return;try{await api("learning/delete",{id:item.id});await loadLearning()}catch(e){alert(e.message)}};
   header.append(title,del);
   const note=document.createElement("p");note.className="muted";note.textContent=item.topic+" · "+item.excerpt.slice(0,190)+"…";
   const source=document.createElement("a");source.href=item.url;source.target="_blank";source.rel="noopener noreferrer";source.textContent="Abrir fonte original";
   row.append(header,note,source);box.append(row);
 }
 if(!data.items.length)box.textContent="Ainda não existem conhecimentos guardados.";
}
$("#learningBtn").onclick=()=>{$("#learningPanel").classList.remove("hidden");$(".sidebar").classList.remove("open");loadLearning().catch(e=>$("#learnFeedback").textContent=e.message)};
$("#closeLearning").onclick=()=>$("#learningPanel").classList.add("hidden");
$("#learningPanel").onclick=e=>{if(e.target===$("#learningPanel"))$("#learningPanel").classList.add("hidden")};
$("#learnForm").onsubmit=async e=>{
 e.preventDefault();const btn=$("#learnSearch");btn.disabled=true;
 $("#learnFeedback").textContent="A pesquisar uma fonte pública autorizada…";
 try{
   const data=await api("learning/search",{topic:$("#learnTopic").value,query:$("#learnQuery").value});
   $("#learnFeedback").textContent=data.results.length+" resultados encontrados · "+data.saved+" novos conhecimentos guardados automaticamente.";
   await loadLearning();
 }catch(err){$("#learnFeedback").textContent=err.message}
 finally{btn.disabled=false}
};

async function loadEvolution(){
 const x=await api("evolution");
 $("#evoSummary").textContent=x.examples+" correções guardadas · modelo ativo: "+x.active;
 const versions=$("#evoVersions");versions.replaceChildren();
 const all=[{model:x.base,status:x.active===x.base?"active":"base"},...x.versions];
 const seen=new Set();
 for(const v of all){
  if(seen.has(v.model))continue;seen.add(v.model);
  const row=document.createElement("div");row.className="model-row";
  const label=document.createElement("strong");label.textContent=v.model+" · "+(v.model===x.active?"Ativo":v.status);
  row.append(label);
  if(v.model!==x.active){
   const btn=document.createElement("button");btn.textContent="Ativar / rollback";
   btn.onclick=async()=>{if(!confirm("Ativar "+v.model+"?"))return;try{await api("evolution/activate",{model:v.model});await loadEvolution();await loadModels()}catch(err){$("#evoFeedback").textContent=err.message}};
   row.append(btn);
  }versions.append(row);
 }
 const box=$("#evoExamples");box.replaceChildren();
 for(const ex of x.examples_list){
  const row=document.createElement("div");row.className="learn-item";
  const title=document.createElement("strong");title.textContent=ex.prompt;
  const answer=document.createElement("p");answer.textContent=ex.correction;
  const remove=document.createElement("button");remove.className="ghost";remove.textContent="Remover";
  remove.onclick=async()=>{try{await api("evolution/delete",{id:ex.id});await loadEvolution()}catch(err){alert(err.message)}};
  row.append(title,answer,remove);box.append(row);
 }
 if(!x.examples_list.length)box.textContent="Ainda não existem correções.";
}
$("#evolutionBtn").onclick=()=>{$("#evoPanel").classList.remove("hidden");$(".sidebar").classList.remove("open");loadEvolution().catch(e=>$("#evoFeedback").textContent=e.message)};
$("#closeEvo").onclick=()=>$("#evoPanel").classList.add("hidden");
$("#evoPanel").onclick=e=>{if(e.target===$("#evoPanel"))$("#evoPanel").classList.add("hidden")};
$("#evoForm").onsubmit=async e=>{
 e.preventDefault();try{await api("evolution/correction",{prompt:$("#evoPrompt").value,correction:$("#evoCorrection").value});
 $("#evoPrompt").value="";$("#evoCorrection").value="";$("#evoFeedback").textContent="Correção guardada.";await loadEvolution()
 }catch(err){$("#evoFeedback").textContent=err.message}
};
$("#exportDataset").onclick=async()=>{
 try{const data=await api("evolution/dataset");
 const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});
 const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="nord-evolution-dataset.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 $("#evoFeedback").textContent="Dataset exportado com "+data.examples.length+" exemplos."
 }catch(e){$("#evoFeedback").textContent=e.message}
};
