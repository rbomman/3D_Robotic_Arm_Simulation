import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  

def dh_matrix(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0,sa,ca,d],[0,0,0,1]], dtype=float)

def fk_chain(a_list, alpha_list, d_list, q_list):
    Ts=[np.eye(4)]; T=np.eye(4)
    for i in range(len(q_list)):
        A=dh_matrix(a_list[i],alpha_list[i],d_list[i],q_list[i]); T=T@A; Ts.append(T)
    return Ts

def rpy_from_R_zyx(R):
    eps=1e-9
    if abs(R[2,0])<1-eps:
        pitch=np.arcsin(-R[2,0]); roll=np.arctan2(R[2,1],R[2,2]); yaw=np.arctan2(R[1,0],R[0,0])
    else:
        pitch=np.pi/2 if R[2,0]<=-1+eps else -np.pi/2; roll=0.0; yaw=np.arctan2(-R[0,1],R[1,1])
    return roll,pitch,yaw

def R_from_rpy_zyx(roll,pitch,yaw):
    cr,sr=np.cos(roll),np.sin(roll); cp,sp=np.cos(pitch),np.sin(pitch); cy,sy=np.cos(yaw),np.sin(yaw)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]); Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]); Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz@Ry@Rx

def so3_log(R):
    cos_theta=(np.trace(R)-1.0)/2.0; cos_theta=np.clip(cos_theta,-1.0,1.0); theta=np.arccos(cos_theta)
    if theta<1e-9: return np.zeros(3)
    oh=(R-R.T)/(2.0*np.sin(theta))
    return theta*np.array([oh[2,1],oh[0,2],oh[1,0]])

def compute_jacobian(Ts):
    n=6; J=np.zeros((6,n)); pe=Ts[-1][:3,3]
    for i in range(1,n+1):
        z=Ts[i-1][:3,2]; pi=Ts[i-1][:3,3]; J[:3,i-1]=np.cross(z,pe-pi); J[3:,i-1]=z
    return J

a_list=np.array([0.0,-0.40,-0.321,0.0,0.0,0.0],dtype=float)
alpha_list=np.array([np.pi/2,0.0,0.0,np.pi/2,-np.pi/2,0.0],dtype=float)
d_list=np.array([0.089,0.0,0.0,0.109,0.094,0.082],dtype=float)
q_min_deg=np.array([-180,-180,-180,-180,-180,-360],dtype=float)
q_max_deg=np.array([180,180,180,180,180,360],dtype=float)

def ik_solve(q_init, target_T, iters=200, lam=2e-2, step=1.0, tol_pos=1e-4, tol_ori=1e-3):
    q=q_init.copy(); I=np.eye(6); q_path=[q.copy()]
    for k in range(iters):
        Ts=fk_chain(a_list,alpha_list,d_list,q); Te=Ts[-1]
        pe=Te[:3,3]; pt=target_T[:3,3]; e_pos=pt-pe
        Re=Te[:3,:3]; Rt=target_T[:3,:3]; R_err=Re.T@Rt; e_ori=so3_log(R_err)
        e=np.hstack([e_pos,e_ori])
        if np.linalg.norm(e_pos)<tol_pos and np.linalg.norm(e_ori)<tol_ori: return q,True,k,q_path
        J=compute_jacobian(Ts); JT=J.T; dq=JT@np.linalg.solve(J@JT+(lam**2)*I,e)
        dq=np.clip(dq,-0.2,0.2); q=q+step*dq
        qd=np.rad2deg(q); qd=np.minimum(q_max_deg,np.maximum(q_min_deg,qd)); q=np.deg2rad(qd); q_path.append(q.copy())
    return q,False,iters,q_path

class ArmViewer:
    def __init__(self, master):
        self.fig=Figure(figsize=(6.5,6.5),dpi=100); self.ax=self.fig.add_subplot(111,projection='3d')
        self.canvas=FigureCanvasTkAgg(self.fig,master=master); self.canvas.get_tk_widget().pack(fill='both',expand=True)
        total_len=float(np.sum(np.abs(a_list))+np.sum(np.abs(d_list))+0.1); self.radius=max(0.5,total_len)
        self.ax.set_xlabel('X [m]'); self.ax.set_ylabel('Y [m]'); self.ax.set_zlabel('Z [m]')
        self._equalize_axes()
        self.triad_len=0.08; self.arm_line=None; self.ee_axes=[]
        self.set_q(np.zeros(6))

    def _equalize_axes(self):
        r=self.radius; self.ax.set_xlim3d([-r,r]); self.ax.set_ylim3d([-r,r]); self.ax.set_zlim3d([0.0,2*r]); self.ax.set_box_aspect([1,1,1])

    def set_q(self, q):
        Ts=fk_chain(a_list,alpha_list,d_list,q); P=np.array([T[:3,3] for T in Ts])
        if self.arm_line is None:
            (self.arm_line,)=self.ax.plot(P[:,0],P[:,1],P[:,2],'-o',linewidth=2,markersize=5)
            base_O=Ts[0][:3,3]; base_R=Ts[0][:3,:3]
            for i in range(3):
                v=base_O+base_R[:,i]*self.triad_len; self.ax.plot([base_O[0],v[0]],[base_O[1],v[1]],[base_O[2],v[2]])
            self.ee_axes=[]; ee_R=Ts[-1][:3,:3]; ee_O=Ts[-1][:3,3]
            for i in range(3):
                v=ee_O+ee_R[:,i]*self.triad_len; (ln,)=self.ax.plot([ee_O[0],v[0]],[ee_O[1],v[1]],[ee_O[2],v[2]]); self.ee_axes.append(ln)
        else:
            self.arm_line.set_data(P[:,0],P[:,1]); self.arm_line.set_3d_properties(P[:,2])
            ee_R=Ts[-1][:3,:3]; ee_O=Ts[-1][:3,3]
            for i in range(3):
                p0=ee_O; p1=ee_O+ee_R[:,i]*self.triad_len
                self.ee_axes[i].set_data([p0[0],p1[0]],[p0[1],p1[1]]); self.ee_axes[i].set_3d_properties([p0[2],p1[2]])
        self.canvas.draw_idle()

class ControlPanel:
    def __init__(self, master, viewer: ArmViewer):
        self.viewer=viewer
        self.q_vars=[tk.DoubleVar(value=0.0) for _ in range(6)]
        self.animate_var=tk.BooleanVar(value=True)

        frm_sliders=ttk.LabelFrame(master,text='Joint Angles (deg)'); frm_sliders.pack(fill='x',padx=10,pady=6)
        self._val_labels=[]
        for i in range(6):
            row=ttk.Frame(frm_sliders); row.pack(fill='x',padx=6,pady=3)
            ttk.Label(row,text=f'q{i+1}').pack(side='left')
            s=ttk.Scale(row,from_=q_min_deg[i],to=q_max_deg[i],variable=self.q_vars[i],orient='horizontal',command=self._on_slider)
            s.pack(side='left',fill='x',expand=True,padx=8)
            lbl=ttk.Label(row,width=6,text='0'); lbl.pack(side='right'); self._val_labels.append(lbl)
            self.q_vars[i].trace_add('write', lambda *_ , i=i: self._on_q_var_changed(i))

        # --- Current EE Pose readout ---
        frm_pose_out = ttk.LabelFrame(master, text='Current End-Effector Pose')
        frm_pose_out.pack(fill='x', padx=10, pady=6)
        self.pos_vars = [tk.StringVar(value="0.000") for _ in range(3)]
        self.ori_vars = [tk.StringVar(value="0.0") for _ in range(3)]
        # Position row
        rowP = ttk.Frame(frm_pose_out); rowP.pack(fill='x', padx=6, pady=2)
        ttk.Label(rowP, text="Position (m):").pack(side='left')
        self.lbl_px = ttk.Label(rowP, textvariable=self.pos_vars[0], width=8); self.lbl_px.pack(side='left', padx=(6,2))
        ttk.Label(rowP, text="x").pack(side='left')
        self.lbl_py = ttk.Label(rowP, textvariable=self.pos_vars[1], width=8); self.lbl_py.pack(side='left', padx=(12,2))
        ttk.Label(rowP, text="y").pack(side='left')
        self.lbl_pz = ttk.Label(rowP, textvariable=self.pos_vars[2], width=8); self.lbl_pz.pack(side='left', padx=(12,2))
        ttk.Label(rowP, text="z").pack(side='left')
        # Orientation row
        rowO = ttk.Frame(frm_pose_out); rowO.pack(fill='x', padx=6, pady=2)
        ttk.Label(rowO, text="ZYX (deg):").pack(side='left')
        self.lbl_rx = ttk.Label(rowO, textvariable=self.ori_vars[0], width=8); self.lbl_rx.pack(side='left', padx=(6,2))
        ttk.Label(rowO, text="roll").pack(side='left')
        self.lbl_ry = ttk.Label(rowO, textvariable=self.ori_vars[1], width=8); self.lbl_ry.pack(side='left', padx=(12,2))
        ttk.Label(rowO, text="pitch").pack(side='left')
        self.lbl_rz = ttk.Label(rowO, textvariable=self.ori_vars[2], width=8); self.lbl_rz.pack(side='left', padx=(12,2))
        ttk.Label(rowO, text="yaw").pack(side='left')

        # Pose entries (target)
        frm_pose=ttk.LabelFrame(master,text='Target Pose'); frm_pose.pack(fill='x',padx=10,pady=6)
        self.ent_x=self._entry(frm_pose,'x [m]:','0.30'); self.ent_y=self._entry(frm_pose,'y [m]:','0.20'); self.ent_z=self._entry(frm_pose,'z [m]:','0.25')
        self.ent_r=self._entry(frm_pose,'roll°:','0'); self.ent_p=self._entry(frm_pose,'pitch°:','20'); self.ent_yaw=self._entry(frm_pose,'yaw°:','40')

        frm_btn=ttk.Frame(master); frm_btn.pack(fill='x',padx=10,pady=6)
        ttk.Button(frm_btn,text='Solve IK',command=self.solve_ik).pack(side='left',expand=True,fill='x',padx=4)
        ttk.Button(frm_btn,text='Example Target',command=self.example_target).pack(side='left',expand=True,fill='x',padx=4)
        ttk.Button(frm_btn,text='Reset',command=self.reset_all).pack(side='left',expand=True,fill='x',padx=4)
        ttk.Checkbutton(master,text='Animate',variable=self.animate_var).pack(anchor='w',padx=14)

        self._push_to_viewer()  # also fills the pose readout

    def _entry(self,parent,label,initial):
        row=ttk.Frame(parent); row.pack(fill='x',padx=6,pady=3)
        ttk.Label(row,text=label,width=8).pack(side='left')
        e=ttk.Entry(row); e.insert(0,initial); e.pack(side='left',fill='x',expand=True)
        return e

    def _on_slider(self,_):
        self._push_to_viewer()

    def _on_q_var_changed(self, i):
        self._val_labels[i].config(text=f'{self.q_vars[i].get():.0f}')
        self._push_to_viewer()

    def _push_to_viewer(self):
        q_deg=np.array([v.get() for v in self.q_vars],dtype=float)
        q=np.deg2rad(q_deg)
        self.viewer.set_q(q)
        # also refresh pose readout
        Ts=fk_chain(a_list,alpha_list,d_list,q)
        ee_O=Ts[-1][:3,3]; ee_R=Ts[-1][:3,:3]
        roll,pitch,yaw=rpy_from_R_zyx(ee_R)
        self.pos_vars[0].set(f'{ee_O[0]:.3f}'); self.pos_vars[1].set(f'{ee_O[1]:.3f}'); self.pos_vars[2].set(f'{ee_O[2]:.3f}')
        self.ori_vars[0].set(f'{np.rad2deg(roll):.1f}'); self.ori_vars[1].set(f'{np.rad2deg(pitch):.1f}'); self.ori_vars[2].set(f'{np.rad2deg(yaw):.1f}')

    def parse_target(self):
        try:
            x=float(self.ent_x.get()); y=float(self.ent_y.get()); z=float(self.ent_z.get())
            r=np.deg2rad(float(self.ent_r.get())); p=np.deg2rad(float(self.ent_p.get())); yaw=np.deg2rad(float(self.ent_yaw.get()))
        except ValueError:
            messagebox.showerror('Invalid input','Please enter numeric values for the target pose.'); return None
        T=np.eye(4); T[:3,:3]=R_from_rpy_zyx(r,p,yaw); T[:3,3]=np.array([x,y,z]); return T

    def set_sliders_from_q(self,q):
        qd=np.rad2deg(q)
        for i,v in enumerate(self.q_vars):
            v.set(float(qd[i]))  # traces -> _push_to_viewer updates pose

    def _animate_path(self,path,idx=0,step=2):
        if idx>=len(path): return
        q=path[idx]
        self.set_sliders_from_q(q)   # updates viewer + pose
        self.viewer.canvas.get_tk_widget().after(10, lambda: self._animate_path(path, idx+step, step))

    def solve_ik(self):
        Tgoal=self.parse_target()
        if Tgoal is None: return
        q0=np.deg2rad(np.array([v.get() for v in self.q_vars],dtype=float))
        q_sol,ok,used,q_path=ik_solve(q0,Tgoal,iters=300,lam=2e-2,step=1.0)
        if self.animate_var.get():
            self._animate_path(q_path)
        else:
            self.set_sliders_from_q(q_sol)

    def example_target(self):
        for ent,val in [(self.ent_x,'0.30'),(self.ent_y,'0.20'),(self.ent_z,'0.25'),(self.ent_r,'0'),(self.ent_p,'20'),(self.ent_yaw,'40')]:
            ent.delete(0,'end'); ent.insert(0,val)
        self.solve_ik()

    def reset_all(self):
        for v in self.q_vars: v.set(0.0)
        for ent,val in [(self.ent_x,'0.30'),(self.ent_y,'0.20'),(self.ent_z,'0.25'),(self.ent_r,'0'),(self.ent_p,'20'),(self.ent_yaw,'40')]:
            ent.delete(0,'end'); ent.insert(0,val)
        self._push_to_viewer()

def main():
    root=tk.Tk(); root.withdraw()
    win_view=tk.Toplevel(); win_view.title('6-DoF Arm — Viewer'); viewer=ArmViewer(win_view)
    win_ctrl=tk.Toplevel(); win_ctrl.title('6-DoF Arm — Controls (with EE Pose)'); ControlPanel(win_ctrl,viewer)
    try: win_view.geometry('+100+80'); win_ctrl.geometry('+900+80')
    except Exception: pass
    root.mainloop()

if __name__=='__main__':
    main()
