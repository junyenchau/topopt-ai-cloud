# Data Generation Script for 64x64 Topology Optimization with Angled Loads

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time # For the timeout safety
import datetime
import requests

TOPIC_NAME = "unetcnn_angled_overnight"
def send_alert(message):
    try:
        requests.post(f"https://ntfy.sh/{TOPIC_NAME}", 
                      data=message.encode(encoding='utf-8'),
                      timeout=5)
    except Exception:
        pass 

class TopOptSolver:
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin

    def optimize(self, force_vector, fixed_dofs):
        nelx, nely = self.nelx, self.nely
        ndof = 2 * (nelx + 1) * (nely + 1)
        
        # Initial Density (0.4 everywhere)
        x = self.volfrac * np.ones(nely * nelx)
        xPhys = x.copy()
        
        # --- 1. SETUP ELEMENT STIFFNESS (KE) ---
        KE = self.lk()
        
        # --- 2. SETUP NODE MAPPING (EDOF) ---
        edofMat = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = elx * nely + ely
                # Nodes (Column-Major)
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array([
                    2*n1, 2*n1+1, 2*n2, 2*n2+1,
                    2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
                ])

        # --- 3. ROBUST INDEX ASSEMBLY ---
        iK = edofMat[:, :, np.newaxis]
        iK = np.repeat(iK, 8, axis=2).flatten()
        jK = edofMat[:, np.newaxis, :]
        jK = np.repeat(jK, 8, axis=1).flatten()

        # Pre-compute Filter
        H, Hs = self.prepare_filter(nelx, nely, self.rmin)
        
        # Free DOFs
        all_dofs = np.arange(ndof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        # Optimization Loop
        loop = 0
        change = 1
        
        # --- TIMEOUT START ---
        start_time = time.time()
        
        # Increased loop limit to 150 for sharper 64x64 results
        while change > 0.01 and loop < 150:
            loop += 1
            
            # TIMEOUT CHECK (Skip if stuck for > 20s)
            if time.time() - start_time > 20.0:
                return None 

            # --- 4. ASSEMBLE GLOBAL K ---
            rho = np.maximum(1e-9, xPhys)
            E_material = rho ** self.penal
            sK = (KE[np.newaxis, :, :] * E_material[:, np.newaxis, np.newaxis]).flatten()
            K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
            
            # --- 5. SOLVE (KU = F) ---
            K_free = K[free_dofs, :][:, free_dofs]
            F_free = force_vector[free_dofs]
            
            try:
                U_free = spsolve(K_free, F_free)
            except Exception:
                return None # Failed solve
            
            if np.any(np.isnan(U_free)): return None

            U = np.zeros(ndof)
            U[free_dofs] = U_free
            
            # --- 6. SENSITIVITY ANALYSIS ---
            u_ele = U[edofMat]
            ce = np.sum(np.dot(u_ele, KE) * u_ele, axis=1)
            dc = -self.penal * (xPhys ** (self.penal - 1)) * ce
            dv = np.ones(nely * nelx)

            # Filter Sensitivity
            numerator = H.dot(x * dc)
            dc[:] = numerator / Hs / np.maximum(0.001, x)

            # --- 7. OPTIMALITY CRITERIA UPDATE ---
            l1, l2, move = 0, 100000, 0.2
            xnew = x.copy()
            
            while (l2 - l1) > 1e-4 * (l1 + l2):
                lmid = 0.5 * (l2 + l1)
                if lmid < 1e-10: lmid = 1e-10
                B = -dc / dv / lmid
                B = np.maximum(1e-10, B)
                xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(B)))))
                if np.sum(xnew) > self.volfrac * nelx * nely:
                    l1 = lmid
                else:
                    l2 = lmid
            
            change = np.max(np.abs(xnew - x))
            x = xnew
            xPhys = xnew
        
        return xPhys.reshape((nelx, nely), order='F')

    def lk(self):
        E, nu = 1.0, 0.3
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
        return KE

    def prepare_filter(self, nelx, nely, rmin):
        n_filter = int(nelx * nely * (2 * (int(np.ceil(rmin)) - 1) + 1)**2)
        iH = np.zeros(n_filter, dtype=int)
        jH = np.zeros(n_filter, dtype=int)
        sH = np.zeros(n_filter)
        k = 0
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = i1 * nely + j1
                imin = max(i1 - (int(np.ceil(rmin)) - 1), 0)
                imax = min(i1 + (int(np.ceil(rmin)) - 1) + 1, nelx)
                jmin = max(j1 - (int(np.ceil(rmin)) - 1), 0)
                jmax = min(j1 + (int(np.ceil(rmin)) - 1) + 1, nely)
                for i2 in range(imin, imax):
                    for j2 in range(jmin, jmax):
                        e2 = i2 * nely + j2
                        dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                        iH[k], jH[k], sH[k] = e1, e2, max(0, rmin - dist)
                        k += 1
        H = coo_matrix((sH[:k], (iH[:k], jH[:k])), shape=(nelx*nely, nelx*nely))
        Hs = np.sum(H, axis=1).flatten()
        return H, Hs

# --- GENERATION FUNCTION ---
def generate_data(num_samples=3000): # TARGET: 3000 Samples
    nelx, nely = 64, 64 # TARGET: 64x64 Resolution
    
    # rmin=1.5 for sharper details
    solver = TopOptSolver(nelx, nely, volfrac=0.4, penal=3.0, rmin=1.5)
    
    inputs = []
    targets = []
    
    print(f"Starting generation of {num_samples} samples (64x64)...")
    start_time = time.time()
    last_alert_time = start_time
    count = 0
    
    while count < num_samples:
        ndof = 2 * (nelx + 1) * (nely + 1)
        force = np.zeros(ndof)
        fixed_dofs = []
        
        # --- FIXED WALL (Left Edge) ---
        input_fixed = np.zeros((nelx, nely))
        for y in range(nely + 1):
            node_idx = 0 * (nely + 1) + y
            fixed_dofs.extend([2 * node_idx, 2 * node_idx + 1])
            if y < nely: input_fixed[0, y] = 1 
            
        # --- MIXED COMPLEXITY LOADS ---
        # 50% Single Load, 50% Double Load
        num_loads = np.random.choice([1, 2], p=[0.5, 0.5])
        
        # Temporary buffers for input channels
        input_force_x = np.zeros((nelx, nely))
        input_force_y = np.zeros((nelx, nely))
        
        for _ in range(num_loads):
            # 1. Random Position (avoid wall)
            load_y = np.random.randint(0, nely + 1)
            load_x = np.random.randint(2, nelx + 1)
            
            # 2. Random Vector (Angle & Magnitude)
            angle = np.random.uniform(0, 2 * np.pi)
            fx = np.cos(angle)
            fy = np.sin(angle)

            # 3. Apply to Physics
            load_node = load_x * (nely + 1) + load_y
            force[2 * load_node] += fx 
            force[2 * load_node + 1] += fy
            
            # 4. Apply to Input Image (Accumulate)
            input_force_x[min(load_x, nelx-1), min(load_y, nely-1)] += fx
            input_force_y[min(load_x, nelx-1), min(load_y, nely-1)] += fy

        # --- SOLVE ---
        result = solver.optimize(force, np.array(fixed_dofs, dtype=int))
        
        # If solve failed (timeout or singularity), retry
        if result is None:
            continue
            
        # --- STORE DATA ---
        # Correct Shape: (64, 64, 3)
        inp = np.zeros((nelx, nely, 3))
        inp[:,:,0] = input_fixed.T
        inp[:,:,1] = input_force_x.T
        inp[:,:,2] = input_force_y.T
        
        inputs.append(inp)
        targets.append(result) 
        
        count += 1

        elapsed = time.time() - start_time
        avg_time = elapsed / count
        remaining = num_samples - count
        est_seconds = remaining * avg_time
        est_str = str(datetime.timedelta(seconds=int(est_seconds)))
        
        print(f"Sample {count}/{num_samples} | ETA: {est_str} left", end='\r')

        # --- SEND ALERT ---
        if time.time() - last_alert_time > 1800 or count % 100 == 0:
            msg = f"Progress: {count}/{num_samples} ({(count/num_samples)*100:.1f}%)\nETA: {est_str}"
            send_alert(msg)
            last_alert_time = time.time()
        
        # --- SAFETY CHECKPOINT (Every 500 samples) ---
        if count % 500 == 0:
            np.savez(f"checkpoint_data_{count}.npz", 
                     inputs=np.array(inputs), 
                     targets=np.array(targets))
            print(f"\n[Checkpoint] Saved {count} samples.")
            
            # --- ADD THIS: Clear memory after saving ---
            inputs = []
            targets = []

    # Final Save
    inputs = np.array(inputs)
    targets = np.array(targets)
    np.savez("64_angled_data_overnight.npz", inputs=inputs, targets=targets)
    print(f"\nSuccess! Saved to 64_angled_data_overnight.npz")

if __name__ == "__main__":
    generate_data(num_samples=2000)