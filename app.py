# ============================================================
# 🎨 Streamlit Interactive 3D-like Blob Poster with Multiple Shapes
# ============================================================

import streamlit as st
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib import transforms
import pandas as pd
import os

# ----------------------------
# Step 1: Blob Functions
# ----------------------------
def blob_circle(center=(0.5,0.5), r=0.2, points=200, wobble=0.2):
    angles = np.linspace(0, 2*math.pi, points)
    radii = r * (1 + wobble * (np.random.rand(points)-0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y

def blob_square(center=(0.5,0.5), r=0.2, points=200, wobble=0.2):
    corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*r
    x,y=[],[]
    for i in range(4):
        start,end = corners[i],corners[i+1]
        xs = np.linspace(start[0],end[0],points//4)+(np.random.rand(points//4)-0.5)*wobble*r
        ys = np.linspace(start[1],end[1],points//4)+(np.random.rand(points//4)-0.5)*wobble*r
        x.extend(xs); y.extend(ys)
    return np.array(x)+center[0], np.array(y)+center[1]

def blob_heart(center=(0.5,0.5), r=0.2, points=200, wobble=0.2):
    t = np.linspace(0, 2*np.pi, points)
    x = 16*np.sin(t)**3
    y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    x = x/40*r + center[0] + (np.random.rand(points)-0.5)*wobble*r
    y = y/40*r + center[1] + (np.random.rand(points)-0.5)*wobble*r
    return x, y

def blob_triangle(center=(0.5,0.5), r=0.2, points=200, wobble=0.2):
    corners = np.array([[0,np.sqrt(3)/3],[-0.5,-np.sqrt(3)/6],[0.5,-np.sqrt(3)/6],[0,np.sqrt(3)/3]])*r
    x,y=[],[]
    for i in range(3):
        start,end = corners[i],corners[i+1]
        xs = np.linspace(start[0],end[0],points//3)+(np.random.rand(points//3)-0.5)*wobble*r
        ys = np.linspace(start[1],end[1],points//3)+(np.random.rand(points//3)-0.5)*wobble*r
        x.extend(xs); y.extend(ys)
    return np.array(x)+center[0], np.array(y)+center[1]

def blob_star(center=(0.5,0.5), r=0.2, points=200, wobble=0.2):
    t = np.linspace(0, 2*np.pi, 5, endpoint=False)
    outer = np.array([[np.cos(a),np.sin(a)] for a in t])*r
    inner = np.array([[np.cos(a+np.pi/5)*0.5,np.sin(a+np.pi/5)*0.5] for a in t])*r
    coords=np.empty((0,2))
    for o,i in zip(outer,inner):
        coords=np.vstack([coords,o,i])
    coords=np.vstack([coords,coords[0]])
    x,y=[],[]
    for i in range(len(coords)-1):
        start,end=coords[i],coords[i+1]
        xs=np.linspace(start[0],end[0],points//len(coords))+(np.random.rand(points//len(coords))-0.5)*wobble*r
        ys=np.linspace(start[1],end[1],points//len(coords))+(np.random.rand(points//len(coords))-0.5)*wobble*r
        x.extend(xs); y.extend(ys)
    return np.array(x)+center[0], np.array(y)+center[1]

def blob(center=(0.5,0.5), r=0.2, points=200, wobble=0.2, shape="circle"):
    if shape=="circle": return blob_circle(center,r,points,wobble)
    elif shape=="square": return blob_square(center,r,points,wobble)
    elif shape=="heart": return blob_heart(center,r,points,wobble)
    elif shape=="triangle": return blob_triangle(center,r,points,wobble)
    elif shape=="star": return blob_star(center,r,points,wobble)
    else: return blob_circle(center,r,points,wobble)

# ----------------------------
# Step 2: Palette
# ----------------------------
def make_palette(k=6, mode="pastel", base_h=0.6):
    cols=[]
    for _ in range(k):
        if mode=="pastel": h=random.random(); s=random.uniform(0.15,0.35); v=random.uniform(0.9,1.0)
        elif mode=="vivid": h=random.random(); s=random.uniform(0.8,1.0); v=random.uniform(0.8,1.0)
        elif mode=="mono": h=base_h; s=random.uniform(0.2,0.6); v=random.uniform(0.5,1.0)
        else: h=random.random(); s=random.uniform(0.3,1.0); v=random.uniform(0.5,1.0)
        cols.append(tuple(hsv_to_rgb([h,s,v])))
    return cols

# ----------------------------
# Step 3: Drawing
# ----------------------------
def draw_poster(n_layers, blob_radius_range, wobble_range, alpha_range, shadow_offset, palette_mode, shape, seed):
    random.seed(seed); np.random.seed(seed)
    palette = make_palette(6, mode=palette_mode)
    fig, ax = plt.subplots(figsize=(6,8))
    ax.axis('off')
    ax.set_facecolor((0.97,0.97,0.97))
    blob_paths=[]

    for _ in range(n_layers):
        radius=random.uniform(*blob_radius_range)
        wobble=random.uniform(*wobble_range)
        center=(random.uniform(0.05,0.95), random.uniform(0.05,0.95))
        x,y=blob(center,r=radius,wobble=wobble,shape=shape)
        vertices=np.column_stack([x,y])
        codes=[Path.MOVETO]+[Path.LINETO]*(len(vertices)-1)
        path=Path(vertices,codes)
        color=random.choice(palette)
        alpha=random.uniform(*alpha_range)
        blob_paths.append((path,color,alpha))

    for path,color,alpha in blob_paths:
        shadow=PathPatch(
            path.transformed(transforms.Affine2D().translate(shadow_offset,-shadow_offset)),
            facecolor='black',edgecolor='none',alpha=alpha*0.15,zorder=-2
        )
        ax.add_patch(shadow)

    for path,color,alpha in blob_paths:
        patch=PathPatch(path,facecolor=color,edgecolor='none',alpha=alpha,zorder=-1)
        ax.add_patch(patch)

    ax.text(0.05,0.95,f"Poster • {palette_mode} • {shape}",fontsize=12,weight="bold",transform=ax.transAxes)
    st.pyplot(fig)

# ----------------------------
# Step 4: Streamlit UI
# ----------------------------
st.title("🎨 Interactive Blob Poster Generator")
st.sidebar.header("Adjust Parameters")

n_layers = st.sidebar.slider("Number of Layers", 3, 20, 8)
blob_radius_range = st.sidebar.slider("Blob Radius Range", 0.05, 0.5, (0.1, 0.3))
wobble_range = st.sidebar.slider("Wobble Range", 0.01, 0.5, (0.05, 0.25))
alpha_range = st.sidebar.slider("Alpha Range", 0.1, 1.0, (0.3, 0.6))
shadow_offset = st.sidebar.slider("Shadow Offset", 0.0, 0.1, 0.02)
palette_mode = st.sidebar.selectbox("Palette Mode", ["pastel","vivid","mono","random"])
shape = st.sidebar.selectbox("Shape", ["circle","square","heart","triangle","star"])
seed = st.sidebar.number_input("Seed", 0, 9999, 0)

if st.sidebar.button("Generate Poster"):
    draw_poster(n_layers, blob_radius_range, wobble_range, alpha_range, shadow_offset, palette_mode, shape, seed)
