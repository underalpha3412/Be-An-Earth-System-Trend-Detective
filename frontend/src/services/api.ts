const API=import.meta.env.VITE_API_URL||'http://127.0.0.1:8000';
export async function api<T>(path:string,init?:RequestInit):Promise<T>{const r=await fetch(`${API}${path}`,{...init,headers:{'Content-Type':'application/json',...(init?.headers||{})}});const body=await r.json().catch(()=>({detail:'Invalid server response'}));if(!r.ok)throw new Error(body.detail||`Request failed (${r.status})`);return body as T}
export const apiBase=API;
