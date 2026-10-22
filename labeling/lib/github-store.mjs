import { AppError,initialState,reconcileState } from './state.mjs';
const STATE_PATH='labeling/data/live.json';
export class GitHubStore {
  constructor(env=process.env, request=fetch) {
    this.token=env.LABEL_GITHUB_TOKEN?.trim();this.kind='github';this.repo=env.LABEL_GITHUB_REPOSITORY || 'ComputerVision-Detector/RADU';
    this.branch=env.LABEL_DATA_BRANCH || 'RADU/labeling-data';this.source=env.LABEL_SOURCE_BRANCH || 'RADU/labeling';this.fetch=request;
    if(!/^[\w.-]+\/[\w.-]+$/.test(this.repo)) throw new AppError(503,'GitHub 저장소 설정을 확인하세요.');
    if(this.branch===this.source || ['main','master'].includes(this.branch)) throw new AppError(503,'라벨 저장 브랜치는 코드·main 브랜치와 구분해 주세요.');
  }
  async request(path,options={}) {
    if(!this.token) throw new AppError(503,'공유 저장소의 서버용 토큰이 없습니다. Vercel의 Production 환경에 LABEL_GITHUB_TOKEN을 등록하고 재배포해야 합니다.','GITHUB_TOKEN_MISSING');
    try { return await this.fetch(`https://api.github.com/repos/${this.repo}${path}`,{
      ...options,headers:{Authorization:`Bearer ${this.token}`,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','Content-Type':'application/json',...options.headers},signal:AbortSignal.timeout(12000)}); }
    catch { throw new AppError(503,'공유 저장소 연결이 지연되고 있습니다. 입력 내용은 유지되며 다시 저장할 수 있습니다.'); }
  }
  failure(response) {
    if(response.status===401) return new AppError(503,'GitHub 토큰이 유효하지 않거나 만료되었습니다. 서버 환경변수를 갱신하고 재배포해야 합니다.','GITHUB_TOKEN_INVALID');
    if(response.status===429 || response.status===403 && (response.headers.get('x-ratelimit-remaining')==='0' || response.headers.get('retry-after'))) return new AppError(503,'GitHub 요청 한도에 도달했습니다. 잠시 뒤 다시 시도하세요. 입력 내용은 유지됩니다.','GITHUB_RATE_LIMITED');
    if(response.status===403) return new AppError(503,'GitHub 저장소 접근이 거부되었습니다. 토큰의 RADU 저장소 Contents 읽기·쓰기 권한과 조직 승인을 확인해야 합니다.','GITHUB_PERMISSION_DENIED');
    if(response.status===404) return new AppError(503,'GitHub 저장소 또는 기준 브랜치에 접근할 수 없습니다. 저장소 설정과 토큰 접근 범위를 확인해야 합니다.','GITHUB_REPOSITORY_UNAVAILABLE');
    if(response.status===422) return new AppError(503,'GitHub 저장 요청이 거부되었습니다. 데이터 브랜치 이름·보호 규칙과 토큰 쓰기 권한을 확인해야 합니다.','GITHUB_WRITE_REJECTED');
    return new AppError(503,`GitHub 저장소 응답을 확인해야 합니다 (${response.status}).`);
  }
  async ensureBranch() {
    let r=await this.request(`/git/ref/heads/${this.branch}`);
    if(r.ok)return;
    if(r.status!==404)throw this.failure(r);
    r=await this.request(`/git/ref/heads/${this.source}`);if(!r.ok)throw this.failure(r);
    const base=await r.json();r=await this.request('/git/refs',{method:'POST',body:JSON.stringify({ref:`refs/heads/${this.branch}`,sha:base.object.sha})});
    // Only an actually existing ref is a successful concurrent initialization.
    if(r.status===422){const existing=await this.request(`/git/ref/heads/${this.branch}`);if(existing.ok)return;throw this.failure(r);}
    if(!r.ok)throw this.failure(r);
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
    return {state:reconcileState(state),version:file.sha};
  }
  async write(state,version,author) {
    await this.ensureBranch();
    const body={branch:this.branch,message:`labels: save ${String(author).replace(/[\r\n]/g,' ').slice(0,40)} [skip ci]`,content:Buffer.from(JSON.stringify(state,null,2)+'\n').toString('base64')};
    if(version)body.sha=version;
    const r=await this.request(`/contents/${STATE_PATH}`,{method:'PUT',body:JSON.stringify(body)});
    if(r.status===409)return false;
    if(r.status===422){
      const error=await r.clone().json().catch(()=>({}));
      if(/sha|already exists|does not match/i.test(error.message || ''))return false;
      throw this.failure(r);
    }
    if(!r.ok)throw this.failure(r);return true;
  }
}
