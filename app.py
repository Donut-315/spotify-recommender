# -*- coding: utf-8 -*-
"""
Spotify 歌曲推荐小程序 - Streamlit 版
基于音频特征（danceability, energy, valence, acousticness, bpm）
使用余弦相似度匹配用户偏好
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

def get_music_platform_url(song_name, artist_name, platform):
    keyword = f"{song_name} {artist_name}"
    encoded = urllib.parse.quote(keyword)
    
    if platform == "QQ音乐":
        # QQ音乐移动端搜索（实测可自动填词）
        return f"https://y.qq.com/m/search/?w={encoded}"
    elif platform == "酷狗音乐":
        # 酷狗音乐移动端搜索（keyword参数有效）
        return f"https://m.kugou.com/search/?keyword={encoded}"
    elif platform == "网易云音乐":
        # 网易云音乐移动端搜索（s参数有效）
        return f"https://music.163.com/m/search?keyword={encoded}"   # 注意用 /m/ 移动版
    elif platform == "Apple Music":
        # Apple Music 网页版搜索（term参数有效）
        return f"https://music.apple.com/cn/search?term={encoded}"
    elif platform == "Spotify":
        # Spotify 移动端搜索（自动填词并显示结果）
        return f"https://open.spotify.com/search/{encoded}"
    else:
        return "#"
# ---------- 页面配置 ----------
st.set_page_config(page_title="Spotify 歌曲推荐", layout="wide")
st.title("🎵 Spotify 智能歌曲推荐系统")
st.markdown("根据你的音乐偏好，推荐最匹配的歌曲（基于音频特征）")


# ---------- 数据加载（缓存，避免重复读取） ----------
@st.cache_data
def load_data():
    df_2025 = pd.read_csv('spotify_wrapped_2025_top50_songs.csv')
    df_alltime = pd.read_csv('spotify_alltime_top100_songs.csv')

    df_2025 = df_2025.rename(columns={'streams_2025_billions': 'streams'})
    df_alltime = df_alltime.rename(columns={'total_streams_billions': 'streams'})

    df_2025['source'] = '2025 Wrapped'
    df_alltime['source'] = 'All-Time'

    df_all = pd.concat([df_2025, df_alltime], ignore_index=True)
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'bpm']
    df_all = df_all.dropna(subset=audio_features).reset_index(drop=True)

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_all[audio_features])

    return df_all, X_scaled, scaler, audio_features


df_all, X_scaled, scaler, audio_features = load_data()

# ---------- 侧边栏：用户偏好设置 ----------
st.sidebar.header("🎛️ 设置你的音乐偏好")

# 推荐来源选择
source_option = st.sidebar.radio(
    "推荐来源",
    options=["两者合并（去重）", "仅 2025 Wrapped", "仅 All-Time"],
    index=0
)

# 偏好特征滑块
st.sidebar.subheader("音频特征偏好")
dance_val = st.sidebar.slider("💃 舞蹈性 (Danceability)", 0.0, 1.0, 0.7, 0.01)
energy_val = st.sidebar.slider("⚡ 能量 (Energy)", 0.0, 1.0, 0.6, 0.01)
valence_val = st.sidebar.slider("😊 积极性 (Valence)", 0.0, 1.0, 0.5, 0.01)
acoustic_val = st.sidebar.slider("🎸 声学性 (Acousticness)", 0.0, 1.0, 0.2, 0.01)
bpm_val = st.sidebar.slider("🎵 节拍 (BPM)", 60, 180, 110, 1)

top_n = st.sidebar.number_input("推荐数量", min_value=1, max_value=20, value=5, step=1)

# 在侧边栏添加平台选择（单选）
st.sidebar.subheader("🎵 选择你的音乐平台")
platform = st.sidebar.radio(
    "跳转目标平台",
    options=["QQ音乐", "酷狗音乐", "网易云音乐", "Apple Music"],
    index=0,  # 默认QQ音乐
    horizontal=True
)

# 是否显示原始特征列
show_features = st.sidebar.checkbox("显示歌曲的音频特征", value=True)


# ---------- 推荐函数 ----------
def get_recommendations(target_features, source_type, top_n=5, deduplicate=True):
    # 根据来源筛选
    if source_type == "仅 2025 Wrapped":
        mask = df_all['source'] == '2025 Wrapped'
        df_sub = df_all[mask].copy()
        X_sub = X_scaled[mask]
    elif source_type == "仅 All-Time":
        mask = df_all['source'] == 'All-Time'
        df_sub = df_all[mask].copy()
        X_sub = X_scaled[mask]
    else:  # 两者合并
        df_sub = df_all.copy()
        X_sub = X_scaled

    if len(df_sub) == 0:
        return pd.DataFrame()

    # 目标向量标准化
    target_arr = np.array([target_features[f] for f in audio_features]).reshape(1, -1)
    target_scaled = scaler.transform(target_arr)

    # 相似度
    sim = cosine_similarity(target_scaled, X_sub)[0]
    df_sub = df_sub.copy()
    df_sub['similarity'] = sim

    # 去重（按歌曲+艺人）
    if deduplicate and source_type == "两者合并（去重）":
        df_sub = df_sub.sort_values('similarity', ascending=False)
        df_sub = df_sub.drop_duplicates(subset=['song_title', 'artist'], keep='first')

    # 排序取 TopN
    df_sub = df_sub.sort_values('similarity', ascending=False).head(top_n)
    return df_sub


# ---------- 主界面：推荐按钮 ----------
if st.sidebar.button("🎧 开始推荐", type="primary", use_container_width=True):
    target = {
        'danceability': dance_val,
        'energy': energy_val,
        'valence': valence_val,
        'acousticness': acoustic_val,
        'bpm': float(bpm_val)
    }

    with st.spinner("正在匹配中..."):
        recs = get_recommendations(target, source_option, top_n, deduplicate=True)

    if recs.empty:
        st.warning("没有找到符合条件的歌曲，请尝试其他来源或调整偏好。")
    else:
        st.success(f"为你推荐 {len(recs)} 首歌曲")

        # 展示结果（使用列布局）
        for i, row in recs.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                  music_url = get_music_platform_url(row['song_title'], row['artist'], platform)
                  st.link_button(
                   label=f"**{row['song_title']}** — {row['artist']}",
                    url=music_url,
                    help=f"点击在 {platform} 中搜索此歌"
                 )
                  st.caption(f"来源: {row['source']}  |  相似度得分: {row['similarity']:.4f}")
                with col2:
                    st.metric("播放量", f"{row['streams']:.2f}B" if row['streams'] < 100 else f"{row['streams']:.0f}B")
                with col3:
                    if show_features:
                        st.caption(f"💃 {row['danceability']:.2f}  ⚡ {row['energy']:.2f}  😊 {row['valence']:.2f}")
                st.divider()

        # 可选：显示详细表格（折叠）
        with st.expander("📊 查看详细数据表格"):
            display_cols = ['song_title', 'artist', 'similarity', 'source', 'streams'] + audio_features
            st.dataframe(recs[display_cols], use_container_width=True)

# ---------- 初始提示 ----------
else:
    st.info("👈 请在左侧侧边栏调整你的音乐偏好，然后点击「开始推荐」")

# ---------- 脚注 ----------
st.markdown("---")
st.caption("基于 Spotify 2025 年度榜和 All-Time 历史榜数据 | 使用余弦相似度匹配音频特征")
