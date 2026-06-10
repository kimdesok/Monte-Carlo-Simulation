import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Monte Carlo Election Simulator",
    page_icon="📊",
    layout="wide"
)

# --- APP TITLE & DESCRIPTION ---
st.title("📊 동일 득표수 투표구 수 찾기 시뮬레이터")
st.markdown("""
**몬테카를로 기법**과 **포와송 분포 트래킹**을 활용 동일한 득표수를 보이는 투표소 갯수를 짐작해보는 시뮬레이션임. 
적용된 변수는 정치적 균질성과 특정 후보의 지지정도이며, 주어진 초기값으로 앱을 실행해보면 (Run =100)일 경우 10곳 남짓 나옴.
""")
st.markdown("---")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("🛠️ 시뮬레이션 변수")
st.sidebar.markdown("지역 투표자의 특성에 따라 조정 필요함")

n_precincts = st.sidebar.slider(
    "투표소 갯수 (N)", 
    min_value=300, max_value=1200, value=1050, step=50,
    help="선거구내 비교 대상인 전체 투표소 갯수."
)

avg_voters = st.sidebar.slider(
    "투표소당 투표수", 
    min_value=500, max_value=5000, value=900, step=100,
    help="각 투표소당 투표자수"
)

avg_support = st.sidebar.slider(
    "일등후보 지지율", 
    min_value=50.0, max_value=100.0, value=90.50, step=0.25,
    help="앞서가는 후보에 대한 지지율."
) / 100.0

support_std = st.sidebar.slider(
    "정치성향의 비슷함 (1-10)", 
    min_value=1.0, max_value=10.0, value=9.0, step=0.5,
    help="동네간 정치적 균질성(개인간으로 해석해도 무방). 0.5는 개인간 매우 다름. 10.0은 거의 동일함."
) / 100.0

n_simulations = st.sidebar.number_input(
    "평균 기대치를 구하기 위한 시뮬레이션 수행 수", 
    min_value=10, max_value=500, value=100, step=10,
    help="How many times the entire election will be re-run to find the statistical average."
)

# --- SIMULATION ENGINE ---
if st.sidebar.button(f"🚀 Run {n_simulations} 가상 투표", type="primary"):
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    match_counts = []
    sample_df_to_show = None
    
    # Run the Monte Carlo Loops
    for sim in range(n_simulations):
        # 1. Simulate varying total turnout per precinct using a normal distribution
        turnouts = np.random.normal(loc=avg_voters, scale=avg_voters * 0.1, size=n_precincts).astype(int)
        turnouts = np.clip(turnouts, 500, avg_voters * 2)  # Prevent negative or unreal numbers
        
        # 2. Simulate varying support rates per precinct using a Bounded Beta Distribution
        # 모멘트 방법(Method of Moments)을 통해 사용자 입력값(평균 지지율, 편차)을 
        # 베타 분포의 Alpha, Beta 매개변수로 변환합니다.
        mu = avg_support
        sigma = max(support_std, 0.005)  # 분모가 0이 되는 현상 방지
        variance = sigma ** 2
        
        # 입력된 편차가 수학적 한계치를 넘을 경우를 대비한 안전장치 (베타 분포 경계선 제어)
        if variance >= mu * (1 - mu):
            variance = mu * (1 - mu) * 0.90
            
        alpha = mu * ((mu * (1 - mu) / variance) - 1)
        beta = (1 - mu) * ((mu * (1 - mu) / variance) - 1)
        
        # np.clip 없이도 0%~100% 사이를 부드럽고 사실적으로 유도하는 지지율 생성
        precinct_rates = np.random.beta(alpha, beta, size=n_precincts)
        
        # 3. Calculate candidate votes (Candidate A vs Remaining Votes)
        votes_a = np.random.binomial(n=turnouts, p=precinct_rates)
        votes_b = turnouts - votes_a
        
        # 4. Generate unique ID strings for combinations (e.g., "3030_1440") to find exact duplicates
        vote_pairs = [f"{a}_{b}" for a, b in zip(votes_a, votes_b)]
        
        # 5. Count identical pairs
        df_pairs = pd.DataFrame({"pair": vote_pairs})
        duplicates = df_pairs['pair'].duplicated(keep=False)
        
        # Number of unique duplicate matching groups
        match_count = df_pairs[duplicates]['pair'].nunique()
        match_counts = np.append(match_counts, match_count)
        
        # Keep the last run's data for UI transparency
        if sim == n_simulations - 1:
            sample_df_to_show = pd.DataFrame({
                "Precinct ID": [f"Precinct_{i+1}" for i in range(n_precincts)],
                "투표수 합": turnouts,
                "후보 1 투표수": votes_a,
                "후보 2 투표수": votes_b,
                "Vote Signature": vote_pairs,
                "동일한 득표가 보이는 투표소?": duplicates
            })
            
            # [FIXED SORTING LOGIC]
            # 1. Bring perfectly matched pairs to the very top (True matches first)
            # 2. Group by Candidate A's votes numerically (Highest to Lowest)
            # 3. Group by Candidate B's votes numerically (Highest to Lowest)
            sample_df_to_show = sample_df_to_show.sort_values(
                by=["동일한 득표가 보이는 투표소?", "후보 1 투표수", "후보 2 투표수"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

           
            
        # Update Progress
        progress_bar.progress((sim + 1) / n_simulations)
        status_text.text(f"Simulating election system environment... {sim + 1}/{n_simulations}")
        
    status_text.empty()
    progress_bar.empty()
    
    # --- METRICS DISPLAY ---
    st.header("📈 시뮬레이션 분석")
    
    mean_matches = np.mean(match_counts)
    max_matches = np.max(match_counts)
    min_matches = np.min(match_counts)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="동일 결과 투표구 수", value=f"{mean_matches:.1f} places")
    with col2:
        st.metric(label="최대 투표구 수", value=f"{int(max_matches)} places")
    with col3:
        st.metric(label="최소 투표구 수", value=f"{int(min_matches)} places")
        
    st.markdown("---")
    
    # --- ADDED SECTION: HISTOGRAM VISUALIZATION ---
    st.subheader("📊 분포 히스토그램: 2026년 6월 3일 선거일 당시 동일 득표 광주전남 투표구 실제 수(10) 기준")
    st.markdown("전체 시뮬레이션에 걸쳐 특정한 동일 투표수를 보이는 투표구가 얼마나 자주 발생하는 지를 보여주는 챠트")
    
    # Create the Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # 1. Plot Empirical Histogram from Monte Carlo Data
    sns.histplot(match_counts, binwidth=1, discrete=True, color="#1f77b4", alpha=0.6, 
                 label="Monte Carlo Simulations", ax=ax, stat="probability")
    
    # 2. Plot Theoretical Poisson Curve Overlaid
    empirical_lambda = mean_matches if mean_matches > 0 else 1.0
    x_theoretical = np.arange(0, max(max_matches + 5, 20))
    y_theoretical = poisson.pmf(x_theoretical, empirical_lambda)
    ax.plot(x_theoretical, y_theoretical, color="#ff7f0e", linestyle="--", marker="o", 
            linewidth=2, label=f"Theoretical Poisson (λ={empirical_lambda:.2f})")
    
    # 3. Add a distinct indicator line for "10 matches" reference point
    ax.axvline(x=10, color="red", linestyle="-", linewidth=2.5, 
               label="Real Election Baseline (10 Matches)")
    
    # Aesthetics
    ax.set_title("Distribution of Identical Vote Count Frequencies", fontsize=12, fontweight='bold')
    ax.set_xlabel("Number of Precints that show the indentical votes", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis='y', alpha=0.3)
    
    # Render in Streamlit
    st.pyplot(fig)
    st.markdown("---")
    
    # --- SCIENTIFIC INTERPRETATION ---
    st.subheader("🔬 수학적 검증")
    prob_12_or_less = poisson.cdf(12, empirical_lambda) * 100
    
    st.info(f"""
    **통계 요약:** 주어진 상황에서, 동일 득표가 나올 수학적 기대치 ($\lambda$)는 **{mean_matches:.2f}** 였음. 이는 1회 선거당 투표소의 기대치를 나타냄.  우연으로 인해 **10** 곳 이하로 기대치가 나올 확률은 **{prob_12_or_less:.1f}%**이었음.  
    히스토그램에서 보듯이 아래 위로 그은 빨간색 직선 (X축 10에 위치)이 히스토그램의 피크 가까이 위치하는 것 보았을 때 시물레이션이 정상적으로 기대할 수 있는 수치임을 알 수 있음.
    """)
    
    # --- RAW DATA VIEW FOR ABSOLUTE TRANSPARENCY ---
    st.markdown("---")
    st.subheader("📂 검증을 위한 실제 시뮬레이션 데이터 (Last Run Sample)")
    st.markdown("아래 스프레드시트는 **우연하게** 발생한 동일 투표수를 가진 투표구 시뮬레이션 결과 중 첫번째 수해 결과를 보여줌. 후보별 동일 득표수를 보이는 투표소 자료를 파랑색으로 하일라이트했으며 후보 1 투표수와 후보 2 투표수가 동일하게 서로 다른 투표소에서 자연적으로 발생할 수 있는 예를 명확히 보여줌. 이상 끝!")

    # Function to color code the rows that are perfectly matched
    def highlight_matches(row):
        if row["동일한 득표가 보이는 투표소?"]:
            return ['background-color: #d4edda; color: #155724'] * len(row)  # Soft green text/background
        return [''] * len(row)
    def bold_matches(row):
        if row["동일한 득표가 보이는 투표소?"]:
            # Candidate A와 Candidate B 득표수 컬럼만 선택해서 굵게 만들고 싶다면 특정 인덱스만 지정할 수도 있지만,
            # 행 전체를 깔끔하게 볼드 처리하는 것이 직관적입니다.
            return ['font-weight: bold; color: #2266FF;'] * len(row)
        if not row["동일한 득표가 보이는 투표소?"]:
            # Candidate A와 Candidate B 득표수 컬럼만 선택해서 굵게 만들고 싶다면 특정 인덱스만 지정할 수도 있지만,
            # 행 전체를 깔끔하게 볼드 처리하는 것이 직관적입니다.
            return ['color: #FFFF11;'] * len(row)
        return [''] * len(row)
    # Render the styled dataframe
    st.dataframe(
        sample_df_to_show.style.apply(bold_matches, axis=1), 
        use_container_width=True
    )
    
else:
    st.warning("👈 사이드바에서 변수를 조절하고 시뮬레이션 결과를 보기위해 버튼 클릭")
