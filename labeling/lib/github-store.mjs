import { AppError,initialState } from './state.mjs';
const STATE_PATH='labeling/data/live.json';
export class GitHubStore {
  constructor(env=process.env, request=fetch) {
    this.token=env.LABEL_GITHUB_TOKEN;this.repo=env.LABEL_GITHUB_REPOSITORY || 'ComputerVision-Detector/RADU';
    this.branch=env.LABEL_DATA_BRANCH || 'RADU/labeling-data';this.source=env.LABEL_SOURCE_BRANCH || 'RADU/labeling';this.fetch=request;
    if(!/^[\w.-]+\/[\w.-]+$/.test(this.repo)) throw new AppError(503,'GitHub 저장소 설정을 확인하세요.');
    if(this.branch===this.source || ['main','master'].includes(this.branch)) throw new AppError(503,'라벨 저장 브랜치는 코드·main 브랜치와 구분해 주세요.');
  }
  async request(path,options={}) {
    if(!this.token) throw new AppError(503,'공유 저장소가 아직 연결되지 않았습니다. 배포 관리자가 Vercel에 LABEL_GITHUB_TOKEN을 설정해야 합니다.');
    try { return await this.fetch(`https://api.github.com/repos/${this.repo}${path}`,{
      ...options,headers:{Authorization:`Bearer ${this.token}`,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','Content-Type':'application/json',...options.headers},signal:AbortSignal.timeout(12000)}); }
    catch { throw new AppError(503,'공유 저장소 연결이 지연되고 있습니다. 입력 내용은 유지되며 다시 저장할 수 있습니다.'); }
  }
  failure(response) {
    if(response.status===401 || response.status===403) return new AppError(503,'GitHub 저장 권한 또는 요청 한도를 확인해야 합니다. 관리자에게 알려 주세요.');
    return new AppError(503,`GitHub 저장소 응답을 확인해야 합니다 (${response.status}).`);
  }
  async ensureBranch() {
    let r=await this.request(`/git/ref/heads/${this.branch}`);
    if(r.ok)return;
    if(r.status!==404)throw this.failure(r);
    r=await this.request(`/git/ref/heads/${this.source}`);if(!r.ok)throw this.failure(r);
    const base=await r.json();r=await this.request('/git/refs',{method:'POST',body:JSON.stringify({ref:`refs/heads/${this.branch}`,sha:base.object.sha})});
    // A concurrent initial writer may have just created the branch.
    if(!r.ok && r.status!==422)throw this.failure(r);
  }
  async read() {
    const path=`/contents/${STATE_PATH}?ref=${encodeURIComponent(this.branch)}`;
    let r=await this.request(path);
    if(r.status===404){
      const source=await this.request(`/git/ref/heads/${this.source}`);
      if(!source.ok)throw this.failure(source);
      return {state:initialState(),version:null};
    }
    if(!r.ok)throw this.failure(r);
    const file=await r.json();let state;
    if(file.encoding==='base64')state=JSON.parse(Buffer.from(file.content,'base64').toString('utf8'));
    else { r=await this.request(path,{headers:{Accept:'application/vnd.github.raw+json'}});if(!r.ok)throw this.failure(r);state=await r.json(); }
    if(state.schema!==1 || !state.items)throw new AppError(503,'공유 라벨 데이터 형식을 확인해야 합니다.');
    return {state,version:file.sha};
  }
  async write(state,version,author) {
    await this.ensureBranch();
    const body={branch:this.branch,message:`labels: save ${String(author).replace(/[\r\n]/g,' ').slice(0,40)} [skip ci]`,content:Buffer.from(JSON.stringify(state,null,2)+'\n').toString('base64')};
    if(version)body.sha=version;
    const r=await this.request(`/contents/${STATE_PATH}`,{method:'PUT',body:JSON.stringify(body)});
    if(r.status===409 || r.status===422)return false;
    if(!r.ok)throw this.failure(r);return true;
  }
}
