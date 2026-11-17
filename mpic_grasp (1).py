"""
mpic_grasp.py
近似复现论文：
    T. Gold et al., "Model Predictive Interaction Control for Force Closure Grasping", CDC 2021

场景（对应论文 IV-B）：
    Allegro 右手用食指 + 拇指对径捏住一个小球。球心保持不动，
    小球先绕竖直轴逆时针转 +PHI_MAX，再顺时针转到 -PHI_MAX，最后转回 0。
    MPC 同时保证：法向力在 [F_MIN, F_MAX] 之内、摩擦锥不被破坏、关节不超限。

运行：
    python mpic_grasp.py          （macOS 上要用  mjpython mpic_grasp.py）
需要：
    pip install mujoco casadi numpy matplotlib
    scene_grasp.xml、right_hand_grasp.xml 和 assets/ 文件夹放在同一个目录

程序结构：
    1. 参数
    2. 读取模型
    3. 抓取模型：测量量、抓取矩阵、接触力 (论文式 8-12)
    4. 期望轨迹 (论文式 6)
    5. MPC 控制器 (论文式 3, 5, 7, 13)
    6. 主循环：仿真 + 动画
    7. 画图
"""
import time
import numpy as np
import casadi as ca
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt


# =====================================================================
# 1. 参数
# =====================================================================
XML_PATH = "scene_grasp_v2.xml"   # v2：半径 15 mm 的小球，抓握点按可转角度优化过

DT       = 0.01    # MPC 采样时间 10 ms                    （论文 IV）
T_HOR    = 0.3     # 预测时域 300 ms                        （论文 IV）
N_HOR    = 10      # 预测时域分成 10 段，每段 30 ms
SIM_TIME = 20.0    # 仿真总时长 [s]                         （论文 Fig.5 为 20 s）

MU           = 0.5          # MPC 中使用的摩擦系数（仿真里实际是 1.0，留余量）
F_MIN, F_MAX = 2.0, 6.0     # 法向力上下限 [N]（论文 [2, 8]；v2 姿态下食指 ffj1 只够产生约 6 N）
U_MAX        = 2.0          # 控制量 u = dq_d/dt 的上下限 [rad/s]   （式 3d）

# 代价函数 (7) 的权重
W_TIP = 1e4    # Q_p ：指尖位置误差
W_JNT = 1.0    # Q_q ：次要目标（食指/拇指末关节保持中位，中指/无名指保持不动）
W_U   = 1e-3   # R   ：控制量正则项

# 期望物体运动（具体怎么组合见第 4 节的 object_ref）
#   转动：绕 ROT_AXIS 转 phi，phi 按路点平滑变化
#   平移：球心相对初始位置的偏移，按路点平滑变化（默认全 0 = 球心不动）
ROT_AXIS     = np.array([0.0, 0.0, 1.0])     # 竖直轴，垂直于两指尖连线
PHI_MAX      = np.deg2rad(50)                # v2 姿态可达约 +55°/-60°，留 5° 余量
WAYPOINT_T   = [0, 1, 5, 7, 13, 15, 19, 20]  # [s]
WAYPOINT_PHI = [0, 0, PHI_MAX, PHI_MAX, -PHI_MAX, -PHI_MAX, 0, 0]
WAYPOINT_SHIFT = [[0, 0, 0]] * len(WAYPOINT_T)   # 每个路点的球心偏移 [m]，例子见 object_ref


# =====================================================================
# 2. 读取模型
# =====================================================================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

NQ = 16   # 手的关节数。顺序：食指 0-3，中指 4-7，无名指 8-11，拇指 12-15

KP = model.actuator_gainprm[:NQ, 0].copy()   # 位置执行器刚度 kp（xml 里 = 1）
KD = model.dof_damping[:NQ].copy()           # 关节阻尼（xml 里 = 0.1），作为 PD 的 D
Q_MIN,  Q_MAX  = model.jnt_range[:NQ, 0].copy(), model.jnt_range[:NQ, 1].copy()
QD_MIN, QD_MAX = model.actuator_ctrlrange[:NQ, 0].copy(), model.actuator_ctrlrange[:NQ, 1].copy()

m_obj = model.body("object").mass[0]
F_O = np.array([0, 0, -m_obj * 9.81, 0, 0, 0])   # 物体力旋量：只考虑重力（论文 IV-B）

TIP_SITES  = [model.site("ff_tip_point").id, model.site("th_tip_point").id]   # 食指、拇指
TIP_BODIES = [model.body("ff_tip").id, model.body("th_tip").id]
OBJ_GEOM   = model.geom("object_geom").id


# =====================================================================
# 3. 抓取模型
# =====================================================================
def skew(r):
    """叉乘矩阵：skew(r) @ f = r x f"""
    return np.array([[0, -r[2], r[1]],
                     [r[2], 0, -r[0]],
                     [-r[1], r[0], 0]])


def rotation(axis, angle):
    """罗德里格斯公式：绕单位轴 axis 转 angle 的旋转矩阵"""
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def measure(data):
    """
    从仿真读取测量值：
        q   : 16 个关节角
        p   : 两个指尖点的位置，2x3（第 0 行食指，第 1 行拇指）
        J_h : 手雅可比，6x16，满足  [dp_ff; dp_th] = J_h @ dq
    """
    q = data.qpos[:NQ].copy()
    p = np.array([data.site_xpos[s] for s in TIP_SITES])
    J_h = np.zeros((6, NQ))
    jacp = np.zeros((3, model.nv))
    for i, s in enumerate(TIP_SITES):
        mujoco.mj_jacSite(model, data, jacp, None, s)
        J_h[3*i:3*i+3, :] = jacp[:, :NQ]
    return q, p, J_h


def grasp_model(p, J_h):
    """
    抓取模型（论文 II、III-D）。手掌固定，所以这里的基坐标系就是世界坐标系。
        t_o : 物体位置 = 两指尖中点（论文 IV-B 也是这样估计的）
        z   : 两个接触法向，沿两指尖连线、指向物体内部，2x3
        P   : 式 (9) 中 [J_h^T; -G] 的 Moore-Penrose 伪逆，6x22
    """
    t_o = p.mean(axis=0)
    z_ff = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])   # 食指 -> 拇指
    z = np.array([z_ff, -z_ff])

    # 抓取矩阵 G：物体受到的力旋量 = G @ F_C（带摩擦点接触，每个手指只传递 3 维力）
    G = np.zeros((6, 6))
    for i in range(2):
        G[0:3, 3*i:3*i+3] = np.eye(3)               # 力直接相加
        G[3:6, 3*i:3*i+3] = skew(p[i] - t_o)         # 力对物体中心产生的力矩

    A = np.vstack([J_h.T, -G])                       # 式 (8) 左边，22x6
    P = np.linalg.inv(A.T @ A) @ A.T                 # 式 (9) 的伪逆
    return t_o, z, P


def contact_forces(P, q, q_d):
    """式 (9)：F_C = P @ [-K_P (q - q_d); F_o]。返回 2x3，每行是一个手指对物体的力"""
    b = np.concatenate([-KP * (q - q_d), F_O])
    return (P @ b).reshape(2, 3)


def mujoco_normal_forces(data):
    """仿真中真实的指尖-小球法向接触力（只用来和模型预测对比画图）"""
    fn = np.zeros(2)
    f6 = np.zeros(6)
    for k in range(data.ncon):
        con = data.contact[k]
        if OBJ_GEOM not in (con.geom1, con.geom2):
            continue
        bodies = (model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2])
        for i, b in enumerate(TIP_BODIES):
            if b in bodies:
                mujoco.mj_contactForce(model, data, k, f6)   # f6[0] 是法向力
                fn[i] += f6[0]
    return fn


# =====================================================================
# 4. 期望轨迹
# =====================================================================
def smooth_path(t, times, values):
    """在路点之间用余弦曲线平滑过渡。values 可以是标量列表，也可以是向量列表"""
    values = np.asarray(values, dtype=float)
    t = np.clip(t, times[0], times[-1])
    k = min(np.searchsorted(times, t, side="right") - 1, len(times) - 2)
    s = (t - times[k]) / (times[k+1] - times[k])   # 0 -> 1
    s = 0.5 - 0.5 * np.cos(np.pi * s)              # 平滑后的 0 -> 1
    return values[k] + (values[k+1] - values[k]) * s


def object_ref(t):
    """
    期望物体位姿（相对初始位姿）：返回 (球心偏移 [m], 旋转矩阵)。
    想让手做别的操作，只需要改这个函数，例如：
      - 绕别的轴转：把 ROT_AXIS 换成别的单位向量（最好垂直于两指尖连线）
      - 边转边移：WAYPOINT_SHIFT = [[0,0,0],[0,0,0],[0,0,0.01],...]（单位 m）
      - 两个轴组合：return shift, rotation(a1, phi1) @ rotation(a2, phi2)
    改完先用 check_reach.py 检查手指够不够得着，再跑 MPC。
    """
    phi = smooth_path(t, WAYPOINT_T, WAYPOINT_PHI)
    shift = smooth_path(t, WAYPOINT_T, WAYPOINT_SHIFT)
    return shift, rotation(ROT_AXIS, phi)


def desired_tips(t, t_o_init, r_o):
    """
    式 (6)：t_fi* = R_o*(t) @ t_fi^o + t_o*(t)
        t_o_init : 初始物体位置
        r_o      : 指尖在物体坐标系下的位置，2x3
    返回 6 维向量 [p_ff*; p_th*]
    """
    shift, R = object_ref(t)
    return ((R @ r_o.T).T + t_o_init + shift).reshape(-1)


# =====================================================================
# 5. MPC 控制器
# =====================================================================
class MPIC:
    """
    模型预测交互控制（论文 III），用 CasADi 的 Opti 搭建，IPOPT 求解。
        状态  x = [q; q_d]   （32 维：实际关节角 + 期望关节角）
        控制  u = dq_d/dt    （16 维）
    """

    def __init__(self, q_ref, w_jnt):
        N = N_HOR
        self.dtau = T_HOR / N
        opti = ca.Opti()

        # ---------- 决策变量 ----------
        self.X = opti.variable(2 * NQ, N + 1)   # 状态轨迹
        self.U = opti.variable(NQ, N)           # 控制轨迹

        # ---------- 参数：每个采样时刻用测量值更新 ----------
        self.x0    = opti.parameter(2 * NQ)     # 当前状态 [q_k; q_d,k]
        self.q_k   = opti.parameter(NQ)         # 当前测得的关节角
        self.p_k   = opti.parameter(6)          # 当前指尖位置 [p_ff; p_th]
        self.J_h   = opti.parameter(6, NQ)      # 手雅可比（时域内视为常数，论文 III-D）
        self.P     = opti.parameter(6, NQ + 6)  # 式 (9) 的伪逆
        self.z     = opti.parameter(3, 2)       # 两个接触法向（每列一个）
        self.p_des = opti.parameter(6, N + 1)   # 时域内各时刻的期望指尖位置

        # ---------- 代价 (7) 与约束 (3b)(3c)(3d) ----------
        cost = 0
        opti.subject_to(self.X[:, 0] == self.x0)            # 初始条件
        for k in range(N):
            x_k, u_k, x_next = self.X[:, k], self.U[:, k], self.X[:, k + 1]

            # 系统动力学 (5)，用显式欧拉法离散
            opti.subject_to(x_next == x_k + self.dtau * self.dynamics(x_k, u_k))
            # 控制量约束 (3d)
            opti.subject_to(opti.bounded(-U_MAX, u_k, U_MAX))

            q, q_d = x_next[:NQ], x_next[NQ:]

            # 代价 (7) = 指尖位置误差 + 关节次要目标 + 控制正则
            p = self.p_k + ca.mtimes(self.J_h, q - self.q_k)   # 正运动学在当前点线性化
            cost += W_TIP * ca.sumsqr(p - self.p_des[:, k + 1])
            cost += ca.sumsqr(ca.DM(np.sqrt(w_jnt)) * (q - q_ref))
            cost += W_U * ca.sumsqr(u_k)

            # 不等式约束 (13)：关节限位
            opti.subject_to(opti.bounded(Q_MIN, q, Q_MAX))
            opti.subject_to(opti.bounded(QD_MIN, q_d, QD_MAX))   # 执行器指令范围
            # 不等式约束 (13)：法向力上下限 + 摩擦锥
            F_C = self.contact_forces(q_d)
            for i in range(2):
                f = F_C[3*i:3*i+3]
                z = self.z[:, i]
                fn = ca.dot(z, f)          # 式 (11) 法向力
                ft = f - z * fn            # 式 (12) 切向力
                opti.subject_to(opti.bounded(F_MIN, fn, F_MAX))
                opti.subject_to(ca.sqrt(ca.sumsqr(ft) + 1e-9) <= MU * fn)

        opti.minimize(cost * self.dtau)    # 积分 ≈ 求和 x 步长
        opti.solver("ipopt",
                    {"expand": True, "print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": 200, "tol": 1e-6})
        self.opti = opti
        self.X_guess = None   # 用上一次的解做热启动
        self.U_guess = None

    def contact_forces(self, q_d):
        """
        式 (9)。论文假设预测时域内物体位姿不变，因此关节角取测量值 q_k，
        接触力只随预测的 q_d 变化。返回 6x1 = [f_ff; f_th]
        """
        b = ca.vertcat(ca.DM(-KP) * (self.q_k - q_d), ca.DM(F_O))
        return ca.mtimes(self.P, b)

    def dynamics(self, x, u):
        """式 (5)：dq = u - K_D^-1 (K_P (q - q_d) + J_h^T F_bar),  dq_d = u"""
        q, q_d = x[:NQ], x[NQ:]
        F_C = self.contact_forces(q_d)
        # 式 (10)：F_bar 只保留接触力的法向分量
        F_bar = ca.vertcat(*[self.z[:, i] * ca.dot(self.z[:, i], F_C[3*i:3*i+3])
                             for i in range(2)])
        dq = u - (ca.DM(KP) * (q - q_d) + ca.mtimes(self.J_h.T, F_bar)) / ca.DM(KD)
        return ca.vertcat(dq, u)

    def solve(self, x0, q_k, p_k, J_h, P, z, p_des):
        """把测量值代入参数，求解一次，返回第一个控制量 u_0"""
        o = self.opti
        o.set_value(self.x0, x0)
        o.set_value(self.q_k, q_k)
        o.set_value(self.p_k, p_k.reshape(-1))
        o.set_value(self.J_h, J_h)
        o.set_value(self.P, P)
        o.set_value(self.z, z.T)
        o.set_value(self.p_des, p_des)

        # 热启动：第一次用"保持当前状态"，之后用上一次的解往前平移一步
        if self.X_guess is None:
            self.X_guess = np.tile(x0[:, None], (1, N_HOR + 1))
            self.U_guess = np.zeros((NQ, N_HOR))
        o.set_initial(self.X, self.X_guess)
        o.set_initial(self.U, self.U_guess)

        try:
            value = o.solve().value
        except RuntimeError:            # 没有完全收敛时，用最后一次迭代的结果
            value = o.debug.value
            print("  [警告] IPOPT 未收敛，使用最后一次迭代结果")

        X_opt = np.array(value(self.X))
        U_opt = np.array(value(self.U))
        self.X_guess = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        self.U_guess = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
        return U_opt[:, 0]


# =====================================================================
# 6. 主循环
# =====================================================================
def rotation_angle(v0, v, axis):
    """向量 v 相对 v0 绕 axis 转过的角度（论文用两指尖连线的转角表示物体转角）"""
    v0 = v0 - axis * (axis @ v0)
    v = v - axis * (axis @ v)
    return np.arctan2(axis @ np.cross(v0, v), v0 @ v)


def draw_targets(viewer, points):
    """在动画中用红色小球标出期望指尖位置"""
    scn = viewer.user_scn
    scn.ngeom = 0
    for pt in points:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.004, 0, 0]), pt, np.eye(3).flatten(),
                            np.array([1, 0, 0, 0.6], dtype=np.float32))
        scn.ngeom += 1


def main():
    # ---- 初始抓握：载入关键帧，保持指令 0.5 s 让接触稳定 ----
    mujoco.mj_resetDataKeyframe(model, data, model.key("grasp").id)
    for _ in range(round(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    q, p, J_h = measure(data)
    q_d = data.ctrl[:NQ].copy()           # 当前期望关节角 = 关键帧里的指令
    t_o_init = p.mean(axis=0)             # 初始物体位置（期望位置 = 它 + object_ref 给的偏移）
    r_o = p - t_o_init                    # 指尖在物体坐标系中的位置（t=0 时两坐标系同向）
    v0 = p[1] - p[0]                      # 初始两指尖连线，用来测量转角

    # 次要目标：食指、拇指的末关节保持中位（论文 IV-B），中指、无名指保持初始姿态
    q_ref = q.copy()
    w_jnt = np.zeros(NQ)
    for j in [3, 15]:                     # ffj3, thj3
        q_ref[j] = 0.5 * (Q_MIN[j] + Q_MAX[j])
        w_jnt[j] = W_JNT
    w_jnt[4:12] = W_JNT

    mpc = MPIC(q_ref, w_jnt)
    n_sub = round(DT / model.opt.timestep)    # 每个 MPC 周期内仿真走几步（10 ms / 2 ms = 5）
    log = {k: [] for k in ["t", "t_o", "phi", "phi_des", "fn", "ft", "fn_sim"]}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = t_o_init
        viewer.cam.distance = 0.3
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25
        t_wall = time.time()

        for step in range(int(SIM_TIME / DT)):
            if not viewer.is_running():
                break
            t = step * DT

            # (1) 测量并计算抓取模型
            q, p, J_h = measure(data)
            t_o, z, P = grasp_model(p, J_h)

            # (2) 预测时域内的期望指尖位置，式 (6)
            p_des = np.column_stack([desired_tips(t + k * mpc.dtau, t_o_init, r_o)
                                     for k in range(N_HOR + 1)])

            # (3) 求解 MPC
            u = mpc.solve(np.concatenate([q, q_d]), q, p, J_h, P, z, p_des)

            # (4) 记录数据：模型预测的接触力 + 仿真中真实的法向力
            F = contact_forces(P, q, q_d)
            log["t"].append(t)
            p_des_now = p_des[:, 0].reshape(2, 3)
            log["t_o"].append(t_o - p_des_now.mean(axis=0))       # 物体位置误差
            log["phi"].append(rotation_angle(v0, p[1] - p[0], ROT_AXIS))
            log["phi_des"].append(rotation_angle(v0, p_des_now[1] - p_des_now[0], ROT_AXIS))
            log["fn"].append([z[i] @ F[i] for i in range(2)])
            log["ft"].append([np.linalg.norm(F[i] - z[i] * (z[i] @ F[i])) for i in range(2)])
            log["fn_sim"].append(mujoco_normal_forces(data))

            # (5) 把 u_0 作用一个采样周期。
            #     论文的 PD 是 K_P(q_d - q) + K_D(dq_d - dq)，而 MuJoCo 里只有
            #     K_P(ctrl - q) - K_D dq，所以指令加上前馈 K_D/K_P * u，两者就完全一样。
            for _ in range(n_sub):
                q_d = np.clip(q_d + u * model.opt.timestep, QD_MIN, QD_MAX)
                data.ctrl[:NQ] = q_d + KD / KP * u
                mujoco.mj_step(model, data)

            # (6) 刷新动画，速度不超过真实时间
            draw_targets(viewer, p_des_now)
            viewer.sync()
            ahead = (t + DT) - (time.time() - t_wall)
            if ahead > 0:
                time.sleep(ahead)
            if step % 100 == 0:
                print(f"t = {t:5.1f} s   phi_des = {np.rad2deg(log['phi_des'][-1]):6.1f}°   "
                      f"phi = {np.rad2deg(log['phi'][-1]):6.1f}°   fn = {np.round(log['fn'][-1], 2)} N")

    plot(log)


# =====================================================================
# 7. 画图（对应论文 Fig.5 和 Fig.3）
# =====================================================================
def plot(log):
    t = np.array(log["t"])
    t_o = np.array(log["t_o"]) * 1000
    fn, ft, fn_sim = np.array(log["fn"]), np.array(log["ft"]), np.array(log["fn_sim"])
    names = ["index", "thumb"]

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 10))
    ax[0].plot(t, t_o)
    ax[0].legend(["x", "y", "z"], loc="upper right")
    ax[0].set_ylabel("object position\nerror [mm]")

    ax[1].plot(t, np.rad2deg(log["phi_des"]), "k--", label="desired")
    ax[1].plot(t, np.rad2deg(log["phi"]), label="measured")
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel("rotation φ [deg]")

    for i in range(2):
        line, = ax[2].plot(t, fn[:, i], label=f"{names[i]} (MPC model)")
        ax[2].plot(t, fn_sim[:, i], ":", color=line.get_color(), label=f"{names[i]} (MuJoCo)")
    ax[2].axhline(F_MIN, color="gray", ls="--")
    ax[2].axhline(F_MAX, color="gray", ls="--")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_ylabel("normal force [N]")

    for i in range(2):
        ax[3].plot(t, ft[:, i] / fn[:, i], label=names[i])
    ax[3].axhline(MU, color="gray", ls="--", label="μ")
    ax[3].legend(loc="upper right")
    ax[3].set_ylabel("‖f_t‖ / f_n")
    ax[3].set_xlabel("time [s]")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
