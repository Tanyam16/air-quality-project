"""Delhi next-day air-quality demo: 5 neural networks forecast PM2.5 and PM10,
then Person 2's hand-written fuzzy rules turn the forecast into a risk class.

Run with:  streamlit run app.py
Needs:     models/ann_models.joblib   data/delhi_clean.csv   membership_initial.json   fuzzy_engine.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import altair as alt

from fuzzy_engine import FuzzySystem, CLASS_NAMES, OUT_SETS, SCORE_CUTS, _mf

HERE = Path(__file__).parent
FEATURES = ['pm25', 'pm10', 'no2', 'temp', 'humidity', 'wind']
LOCKDOWN = pd.Timestamp('2020-03-25')
COLORS = {0: '#2E8B7F', 1: '#D39A2C', 2: '#B03A30'}
MEANING = {0: 'Air is mostly clean to satisfactory.', 1: 'Air is moderate to poor. Sensitive people should take care.',
           2: 'Air is very poor to severe. Avoid time outdoors.'}

st.set_page_config(page_title='Delhi air quality, a day ahead', layout='wide')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp textarea { font-family: 'Public Sans', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
h1, h2, h3 { font-family: 'Newsreader', serif !important; font-weight: 600 !important; letter-spacing: -0.01em; }
.block-container { max-width: 1120px; padding-top: 4.5rem; }
.hero-title { font-family: 'Newsreader', serif; font-size: 2.5rem; font-weight: 600; line-height: 1.1; margin: 0; }
.hero-sub { opacity: .8; max-width: 62ch; margin: .5rem 0 1.4rem 0; line-height: 1.5; }
.result { background: #fff; color: #1F2B33; border: 1px solid #D5DCDA; border-radius: 6px; padding: 1.4rem 1.6rem; }
.result .label { color: #3E4C55; font-size: .9rem; }
.result .klass { font-family: 'Newsreader', serif; font-size: 3.6rem; font-weight: 600; line-height: 1; margin: .15rem 0 .2rem 0; }
.result .meaning { color: #2A373F; margin-bottom: 1.1rem; }
.band { position: relative; height: 14px; border-radius: 7px; overflow: visible; display: flex; margin-top: .6rem; }
.band div { height: 100%; }
.marker { position: absolute; top: -7px; width: 4px; height: 28px; background: #1F2B33; border-radius: 2px; }
.ticks { position: relative; height: 1.2rem; font-size: .78rem; color: #3E4C55; margin-top: .5rem; }
.ticks span { position: absolute; transform: translateX(-50%); }
.facts { display: flex; gap: 2.2rem; flex-wrap: wrap; margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid #E3E8E6; }
.facts .n { font-family: 'Newsreader', serif; font-size: 1.9rem; font-weight: 600; line-height: 1.1; color: #1F2B33; }
.facts .u { color: #3E4C55; font-size: .85rem; }
.note { opacity: .8; font-size: .88rem; line-height: 1.45; }
.fired { display: grid; grid-template-columns: 4.2rem 1fr; gap: .2rem .8rem; align-items: center; margin: .35rem 0; }
.fired .bar { height: 8px; border-radius: 4px; background: #D5DCDA; }
.fired .bar i { display: block; height: 100%; border-radius: 4px; }
.flow { display: flex; gap: .5rem; flex-wrap: wrap; margin: .4rem 0 1.2rem 0; }
.flow .box { flex: 1 1 130px; background: #fff; color: #1F2B33; border: 1px solid #D5DCDA; border-radius: 6px; padding: .65rem .8rem; font-size: .86rem; line-height: 1.3; opacity: .5; }
.flow .box.on { background: #2B4C63; color: #fff; border-color: #2B4C63; opacity: 1; }
.flow .box b { display: block; font-family: 'Newsreader', serif; font-size: 1.02rem; font-weight: 600; }
.timeline { display: flex; height: 46px; border-radius: 6px; overflow: hidden; margin: .6rem 0 .3rem 0; }
.timeline div { position: relative; display: flex; align-items: center; justify-content: center; color: #fff; font-size: .85rem; font-weight: 500; }
.timeline .lock { position: absolute; top: 0; bottom: 0; width: 3px; background: #fff; }
.tl-labels { display: flex; font-size: .8rem; opacity: .85; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
def need(path, hint):
    if not (HERE / path).exists():
        st.error(f'Missing file: {path}. {hint}')
        st.stop()
    return HERE / path


@st.cache_resource
def load_models():
    p = need('models/ann_models.joblib', 'Copy ann_models.joblib from the ANN notebook into the models folder.')
    pack = joblib.load(p)
    return pack['models'], pack['x_scaler'], pack['y_scaler']


@st.cache_data
def load_windows():
    """Rebuild the 3-day windows exactly as the ANN notebook did, to replay real days."""
    p = need('data/delhi_clean.csv', 'Copy delhi_clean.csv into the data folder.')
    data = pd.read_csv(p, parse_dates=['date']).set_index('date').sort_index()
    vals, rows = data[FEATURES].values, []
    for i in range(2, len(data) - 1):
        w, t = vals[i - 2:i + 1], vals[i + 1, [0, 1]]
        if np.isnan(w).any() or np.isnan(t).any():
            continue
        rows.append({'today': data.index[i], 'window': w, 'true_pm25': t[0], 'true_pm10': t[1],
                     'true_class': data['risk'].iloc[i + 1]})
    W = pd.DataFrame(rows)
    n = len(W); i_tr, i_va = int(n * .6), int(n * .8)
    W['period'] = ['ann_train'] * i_tr + ['fuzzy_fit'] * (i_va - i_tr) + ['test'] * (n - i_va)
    return W


@st.cache_data
def load_raw():
    p = need('data/delhi_clean.csv', 'Copy delhi_clean.csv into the data folder.')
    return pd.read_csv(p, parse_dates=['date']).set_index('date').sort_index()


HAND_RULES = [
    ({'pm25': 'high', 'pm10': 'high'}, 2), ({'pm25': 'high', 'wind': 'calm'}, 2),
    ({'pm25': 'high', 'season': 'winter'}, 2), ({'pm25': 'medium', 'wind': 'calm', 'season': 'winter'}, 2),
    ({'pm25': 'medium', 'humidity': 'humid', 'season': 'winter'}, 2),
    ({'pm25': 'medium', 'pm10': 'high', 'season': 'winter'}, 2),
    ({'pm25': 'medium', 'unc': 'high', 'season': 'winter'}, 2),
    ({'pm25': 'high', 'wind': 'strong', 'season': 'other'}, 1), ({'pm25': 'medium', 'pm10': 'medium'}, 1),
    ({'pm25': 'medium', 'wind': 'strong'}, 1), ({'pm25': 'low', 'pm10': 'medium'}, 1),
    ({'pm25': 'low', 'unc': 'high'}, 1), ({'pm25': 'low', 'season': 'winter'}, 1),
    ({'pm25': 'low', 'season': 'transition'}, 1), ({'pm25': 'low', 'pm10': 'low', 'season': 'other'}, 0),
    ({'pm25': 'low', 'pm10': 'low', 'unc': 'low'}, 0),
]


@st.cache_resource
def load_fuzzy():
    p = need('membership_initial.json', 'Copy membership_initial.json from the fuzzy notebook output.')
    v = json.load(open(p))['variables']
    return FuzzySystem({k: {l: (kind, params) for l, (kind, params) in labs.items()} for k, labs in v.items()}, HAND_RULES)


NAMES = {'pm25': 'forecast PM2.5', 'pm10': 'forecast PM10', 'unc': 'forecast spread', 'wind': 'wind',
         'humidity': 'humidity', 'season': 'season'}


def rule_plain(ant, c):
    cond = ' and '.join(f'{NAMES[v]} is {lab}' for v, lab in ant.items())
    return f'If {cond}, then {CLASS_NAMES[c]}'


# ----------------------------------------------------------------------------- model calls
def run_ann(window, today):
    models, xs, ys = load_models()
    m = today.month
    x = np.concatenate([window.flatten(), [np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]])[None, :]
    xs_ = xs.transform(x)
    preds = np.array([ys.inverse_transform(net.predict(xs_)) for net in models])    # (5, 1, 2)
    return preds.mean(0)[0], float(preds.std(0)[0, 0]), preds[:, 0, 0]


def run_fuzzy(pm25, pm10, unc, wind, hum, season_month):
    fz = load_fuzzy()
    data = {'pm25': [pm25], 'pm10': [pm10], 'unc': [unc], 'wind': [wind], 'humidity': [hum], 'season': [season_month]}
    return fz, data, int(fz.predict_vote(data)[0]), float(fz.mamdani_score(data)[0]), fz.explain(data, 0, top=8)


def band_html(score):
    s = float(np.clip(score, 0, 100))
    return (f'<div class="band"><div style="width:35%;background:{COLORS[0]}"></div>'
            f'<div style="width:30%;background:{COLORS[1]}"></div><div style="width:35%;background:{COLORS[2]}"></div>'
            f'<div class="marker" style="left:calc({s}% - 2px)"></div></div>'
            '<div class="ticks"><span style="left:0%">0</span><span style="left:35%">35</span>'
            '<span style="left:65%">65</span><span style="left:100%">100</span></div>')


# ----------------------------------------------------------------------------- page
st.markdown('<p class="hero-title">Delhi air quality, a day ahead</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Five neural networks read the last three days of pollution and weather and forecast '
            'tomorrow\'s PM2.5 and PM10. A set of readable fuzzy rules then turns that forecast into a risk class, '
            'and shows which rules decided it.</p>', unsafe_allow_html=True)

tab_demo, tab_how, tab_res = st.tabs(['Forecast', 'How it works', 'Results'])

# ============================================================================= Forecast
with tab_demo:
    W = load_windows()
    left, right = st.columns([5, 6], gap='large')

    with left:
        mode = st.radio('What do you want to try?', ['Replay a real day', 'Enter your own readings'], horizontal=True)
        dates = pd.DatetimeIndex(W['today'])
        if mode == 'Replay a real day':
            ex = st.columns(3)
            test = W[W.period == 'test']
            picks = {'Worst smog day': test.loc[test.true_pm25.idxmax(), 'today'],
                     'A clean day': test.loc[test[test.true_pm25 < 50].true_pm25.idxmin(), 'today'] if (test.true_pm25 < 50).any() else dates[-1],
                     'During lockdown': pd.Timestamp('2020-04-20')}
            for col, (label, d) in zip(ex, picks.items()):
                if col.button(label, width='stretch'):
                    st.session_state['today'] = d.date()
            default = st.session_state.get('today', picks['Worst smog day'].date())
            today = pd.Timestamp(st.date_input('Today is', value=default, min_value=dates.min().date(),
                                               max_value=dates.max().date()))
        else:
            today = pd.Timestamp(st.date_input('Today is', value=pd.Timestamp('2019-12-01').date(),
                                               min_value=pd.Timestamp('2015-01-01').date(),
                                               max_value=pd.Timestamp('2023-12-31').date(),
                                               help='The model only uses the month of this date, to account for the smog season.'))
        match = W[W.today == today]
        if len(match):
            row, window0 = match.iloc[0], match.iloc[0]['window']
        else:
            row, window0 = None, W.iloc[-1]['window']
            if mode == 'Replay a real day':
                st.info('That day has no complete three-day record. Pick another date.')
                st.stop()
            st.caption('No record for this date, so the grid starts from the latest real days. Edit any value.')

        st.markdown('**Last three days of readings** (edit any value to ask "what if?")')
        grid = pd.DataFrame(window0, columns=['PM2.5', 'PM10', 'NO2', 'Temp (°C)', 'Humidity (%)', 'Wind (km/h)'],
                            index=['Two days ago', 'Yesterday', 'Today']).round(1)
        grid = st.data_editor(grid, key=f'grid_{mode}_{today.date()}', width='stretch',
                              column_config={c: st.column_config.NumberColumn(format='%.1f') for c in grid.columns})
        if grid.isna().any().any():
            st.warning('Fill every cell to get a forecast.')
            st.stop()
        st.markdown(f'<p class="note">PM values are in µg/m³. The forecast is for '
                    f'{(today + pd.Timedelta(days=1)).strftime("%d %b %Y")}.</p>', unsafe_allow_html=True)

    window = grid.values.astype(float)
    (p25, p10), unc, per_net = run_ann(window, today)
    wind_today, hum_today = window[-1, 5], window[-1, 4]
    fz, fdata, klass, score, fired = run_fuzzy(p25, p10, unc, wind_today, hum_today, (today + pd.Timedelta(days=1)).month)
    score_class = int(fz.score_to_class(score))

    with right:
        st.markdown(
            f'<div class="result"><div class="label">Risk tomorrow</div>'
            f'<div class="klass" style="color:{COLORS[klass]}">{CLASS_NAMES[klass]}</div>'
            f'<div class="meaning">{MEANING[klass]}</div>'
            f'<div class="label">Risk score {score:.0f} out of 100</div>{band_html(score)}'
            f'<div class="facts"><div><div class="n">{p25:.0f}</div><div class="u">PM2.5 forecast, µg/m³</div></div>'
            f'<div><div class="n">{p10:.0f}</div><div class="u">PM10 forecast, µg/m³</div></div>'
            f'<div><div class="n">±{unc:.1f}</div><div class="u">disagreement between the 5 networks</div></div></div></div>',
            unsafe_allow_html=True)
        if score_class != klass:
            st.markdown('<p class="note">The class comes from the rules\' strongest vote and the score is a blended view, '
                        'so close to a boundary they can differ by one step.</p>', unsafe_allow_html=True)

        unc_high = float(fdata and _mf(*fz.variables['unc']['high'], np.array([unc]))[0])
        st.markdown(
            f'<p class="note">The networks {"disagree noticeably" if unc_high > 0.5 else "mostly agree"} on this day '
            f'(their forecasts range from {per_net.min():.0f} to {per_net.max():.0f}). The spread is a modest reliability '
            f'signal: it also grows when pollution is high.</p>', unsafe_allow_html=True)

        if row is not None and mode == 'Replay a real day' and np.allclose(window, window0, atol=0.06):
            actual = int(row.true_class) if not pd.isna(row.true_class) else None
            txt = f'On the day itself, PM2.5 was {row.true_pm25:.0f} and PM10 was {row.true_pm10:.0f}'
            txt += f', and the recorded class was {CLASS_NAMES[actual]}.' if actual is not None else '.'
            st.markdown(f'**Actual outcome.** {txt}')
            if row.period == 'ann_train':
                st.warning('The networks were trained on this day, so the forecast looks better than it would on a new day. '
                           'Pick a date after 19 May 2018 for a fair check.')
            if today + pd.Timedelta(days=1) >= LOCKDOWN:
                st.warning('This falls in the COVID lockdown. The networks had never seen air this clean and tend to over-predict.')

        st.markdown('**Rules that fired**')
        if not fired:
            st.markdown('No rule matched, so the system falls back to Moderate.')
        for text, strength in fired:
            c = int([k for k, nme in enumerate(CLASS_NAMES) if text.endswith(nme)][0])
            ant = dict(part.split(' is ') for part in text[3:text.index(' THEN')].split(' AND '))
            st.markdown(f'<div class="fired"><div>{strength:.2f}</div><div>{rule_plain(ant, c)}'
                        f'<div class="bar"><i style="width:{strength * 100:.0f}%;background:{COLORS[c]}"></i></div></div></div>',
                        unsafe_allow_html=True)

        with st.expander('How each input was read by the rules'):
            rows_ = []
            for v, val in [('pm25', p25), ('pm10', p10), ('unc', unc), ('wind', wind_today), ('humidity', hum_today),
                           ('season', (today + pd.Timedelta(days=1)).month)]:
                parts = [f'{lab} {float(_mf(k, p, np.array([val]))[0]):.2f}' for lab, (k, p) in fz.variables[v].items()
                         if float(_mf(k, p, np.array([val]))[0]) > 0.005]
                rows_.append({'Input': NAMES[v].capitalize(), 'Value': f'{val:.1f}' if v != 'season' else f'month {int(val)}',
                              'Degree of membership': ', '.join(parts)})
            st.dataframe(pd.DataFrame(rows_), hide_index=True, width='stretch')

# ============================================================================= How it works
def flow_strip(active):
    boxes = [('Last 3 days', 'pollution, weather and the month'), ('5 neural networks', 'each makes a forecast'),
             ('Forecast and spread', 'their average, and how far apart they are'), ('Fuzzy rules', '16 readable IF-THEN rules'),
             ('Risk class', 'Low, Moderate or Severe, plus the reasons')]
    return '<div class="flow">' + ''.join(
        f'<div class="box{" on" if i in active else ""}"><b>{t}</b>{d}</div>' for i, (t, d) in enumerate(boxes)) + '</div>'


def points(*items):
    st.markdown('\n'.join(f'- {i}' for i in items))


with tab_how:
    st.header('How it works')
    a_, b_, c_ = st.columns(3, gap='large')
    a_.subheader('1. Forecast')
    a_.write('Five small neural networks each read the last three days of PM2.5, PM10, NO2, temperature, humidity and wind, '
             'plus the month. The average of their answers is tomorrow\'s forecast. How far apart they are tells us how much to trust it.')
    b_.subheader('2. Read it with fuzzy rules')
    b_.write('Instead of hard cutoffs, a value can be partly "medium" and partly "high". Sixteen hand-written rules, such as '
             '"if forecast PM2.5 is high and it is winter, then Severe", each fire to some degree. The strongest vote sets the class.')
    c_.subheader('3. Explain it')
    c_.write('The rules that fired are listed with their strength, so every answer can be traced back to plain-language reasons.')

    st.markdown('**Walk through each stage**')
    stage = st.radio('Stage', ['1. The neural networks', '2. The fuzzy rules', '3. Putting it together'],
                     horizontal=True, label_visibility='collapsed', key='how_stage')
    st.markdown(flow_strip({'1': {0, 1, 2}, '2': {2, 3, 4}, '3': {0, 1, 2, 3, 4}}[stage[0]]), unsafe_allow_html=True)

    # ------------------------------------------------------------------ stage 1
    if stage.startswith('1'):
        st.subheader('Stage 1: forecast tomorrow\'s pollution')
        st.write('The networks learn from Delhi\'s past to predict tomorrow\'s PM2.5 and PM10, and say how much to trust each forecast.')
        m1, m2, m3 = st.columns(3)
        m1.metric('Days of Delhi data', '2,009')
        m1.caption('1 Jan 2015 to 1 Jul 2020')
        m2.metric('Fewer errors than "tomorrow = today"', '12%')
        m2.caption('PM2.5 error 22.3 vs 25.2 (validation)')
        m3.metric('Error on the worst 10% of days', '60.2')
        m3.caption('Best of the three methods, vs 64.0 for "tomorrow = today" (test)')

        t1, t2, t3, t4, t5, t6 = st.tabs(['Data', 'Inputs', 'Split', 'Model', 'Uncertainty', 'Result'])

        with t1:
            points('Joined Delhi\'s daily pollution (CPCB, via Kaggle) with daily weather (Open-Meteo).',
                   'The weather is a model reconstruction, not a ground-station reading.')
            st.markdown('**What we checked**')
            st.dataframe(pd.DataFrame({
                'Check': ['Repeated dates', 'Skipped calendar days', 'Negative values', 'Blank values',
                          'PM2.5 higher than PM10', 'Extreme values'],
                'Why it matters': ['A repeated day would be counted twice', 'Windows need consecutive days',
                                   'Concentrations cannot be negative', 'Gaps break the 3-day windows',
                                   'PM2.5 is part of PM10, so this should not happen', 'Could be sensor errors'],
                'Result': ['0', '0', '0', 'PM2.5 0.1%, NO2 0.1%, PM10 3.8%, AQI category 0.5%',
                           '16 days (about 0.8%)', 'PM2.5 up to 685 on real smog days']}),
                hide_index=True, width='stretch')
            st.markdown('**Cleaning**')
            points('Short gaps (3 days or fewer) were filled with a straight line. The two long PM10 gaps (7 and 58 days) were left blank.',
                   'Extreme smog days were kept, because they are the days that matter.')
            raw = load_raw()
            view = st.radio('Explore the data', ['Average PM2.5 by month', 'Average PM2.5 by year', 'Days in each risk class'],
                            horizontal=True, key='how_explore')
            if view == 'Average PM2.5 by month':
                mon = raw['pm25'].groupby(raw.index.month).mean().round(0)
                mon.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                st.bar_chart(mon.rename('PM2.5, µg/m³'), sort=False, height=260)
                points('Winter is far worse: November peaks at about 238, July and August are the cleanest at about 45.',
                       'This is why the month is a model input.')
            elif view == 'Average PM2.5 by year':
                yr = raw['pm25'].groupby(raw.index.year).mean().round(0)
                yr.index = yr.index.astype(str)
                st.bar_chart(yr.rename('PM2.5, µg/m³'), sort=False, height=260)
                points('2020 is lower, partly because of the COVID lockdown.',
                       'The data also stops in July, before the smog season.')
            else:
                cls = raw['risk'].value_counts().sort_index()
                cls.index = ['Low', 'Moderate', 'Severe']
                st.bar_chart(cls.rename('Days'), sort=False, height=260)
                points('Low days are rare in Delhi (about 9%).', 'Severe days are common (about 38%).')
            points('Yesterday\'s PM2.5 predicts today\'s with a correlation of 0.87.',
                   'So "tomorrow equals today" is a strong baseline that the networks have to beat.')

        with t2:
            points('Each row says: "here is what the last 3 days looked like, so what happened tomorrow?"')
            st.dataframe(pd.DataFrame({'Part': ['Last 3 days of readings', 'Month of the year', 'Total inputs', 'Answers the model learns'],
                                       'Numbers': [18, 2, 20, 2],
                                       'Detail': ['Two days ago, yesterday and today: PM2.5, PM10, NO2, temperature, humidity, wind',
                                                  'The month as a sine and a cosine value',
                                                  '', 'Tomorrow\'s PM2.5 and tomorrow\'s PM10']}),
                         hide_index=True, width='stretch')
            points('The month is two numbers so that December and January sit side by side, not 11 months apart.',
                   'Result: 1,935 rows. 71 windows were lost to the PM10 gaps (about 3.5%).')

        with t3:
            points('The days are split by time, not randomly, so the model is always tested on days after the ones it learned from.',
                   'A random split would put 5 November in training and 4 and 6 November in the test, which leaks the answer.')
            st.markdown('<div class="timeline"><div style="width:60%;background:#2B4C63">Training, 1,161 days</div>'
                        '<div style="width:20%;background:#2E8B7F">Validation, 387</div>'
                        '<div style="width:20%;background:#B07D1B">Test, 387<span class="lock" style="left:74.4%"></span></div></div>'
                        '<div class="tl-labels"><div style="width:60%">4 Jan 2015 to 19 May 2018</div>'
                        '<div style="width:20%">20 May 2018 to 10 Jun 2019</div><div style="width:20%">11 Jun 2019 to 1 Jul 2020</div></div>',
                        unsafe_allow_html=True)
            points('The white line in the test block is 25 March 2020, when the lockdown began: 288 test days before it and 99 during it.',
                   'The test days are used once, at the end.',
                   'Each period contains a full winter, so every stage is judged on the hard season.')
            st.markdown('**Scaling**')
            points('Inputs were put on a similar range (PM10 is in the hundreds, wind in single digits).',
                   'The scalers were fitted on the training days only, so nothing from later days leaks in.')

        with t4:
            st.markdown('**1. The numbers to beat**')
            points('Two simple methods were measured first (validation period, error in µg/m³, lower is better).')
            st.markdown("| Method | PM2.5 error | PM10 error | PM2.5 R² | Winter PM2.5 error |\n|---|---|---|---|---|\n"
                        "| 'Tomorrow = today' | 25.2 | 46.5 | 0.72 | 49.7 |\n"
                        "| Linear regression (straight-line formula) | 22.2 | 40.0 | 0.80 | 41.5 |")
            st.markdown('**2. Tuning the network**')
            points('We tried different network sizes and a penalty against memorising (alpha) on the validation days.')
            tune = pd.DataFrame([
                ('32 neurons, alpha 3.0', 22.8), ('16 neurons, alpha 1.0 (chosen)', 22.9), ('16 neurons, alpha 3.0', 23.0),
                ('16 neurons, alpha 0.001', 23.4), ('16 neurons, alpha 0.1', 23.4), ('8 neurons, any alpha', 23.5),
                ('32 neurons, alpha 1.0', 23.5), ('32 neurons, alpha 10', 23.7), ('16 neurons, alpha 10', 24.0),
                ('Two layers (16 and 8), alpha 1.0', 24.7), ('Two layers (16 and 8), alpha 0.001', 26.0),
                ('Two layers (16 and 8), alpha 0.1', 26.1)], columns=['Setting', 'PM2.5 error'])
            st.dataframe(tune, hide_index=True, width='stretch',
                         column_config={'PM2.5 error': st.column_config.ProgressColumn(
                             'PM2.5 error (lower is better)', format='%.1f', min_value=20, max_value=27)})
            points('The first network (two layers of 32 and 16) scored 25.3, no better than "tomorrow = today".',
                   'It had about as many weights as training rows, so it memorised the training days.',
                   'Smaller networks with more regularisation did better.',
                   'Chosen: 16 neurons, one layer, alpha 1.0. The 32-neuron setting ties within the noise, so the smaller one won.',
                   'We stopped tuning there, because more would be chasing noise.')
            st.markdown('**3. The final model against the other methods**')
            points('The final model is the average of 5 copies of the chosen network.')
            st.markdown("| Method | PM2.5 error | PM10 error | PM2.5 RMSE | PM2.5 R² | Winter PM2.5 error |\n|---|---|---|---|---|---|\n"
                        "| 'Tomorrow = today' | 25.2 | 46.5 | 41.4 | 0.72 | 49.7 |\n"
                        "| Linear regression | 22.2 | 40.0 | 35.4 | 0.80 | 41.5 |\n"
                        "| First network (untuned) | 25.3 | 45.7 | 38.0 | 0.77 | 41.6 |\n"
                        "| **5-network average (final model)** | **22.3** | **41.7** | **35.1** | **0.80** | **40.0** |")
            points('The final model ties linear regression on PM2.5 (22.3 vs 22.2) and has the best winter error (40.0).',
                   'It is slightly worse for PM10 (41.7 vs 40.0).',
                   'Both beat "tomorrow = today".')

        with t5:
            points('One network gives one forecast and no sense of how reliable it is.',
                   'So we trained 5 copies from different random starts.',
                   'The average is the forecast. How much they disagree is the uncertainty.')
            case = st.radio('Try an example (illustrative numbers)', ['The networks agree', 'The networks disagree'],
                            horizontal=True, key='how_unc')
            vals = np.array([240, 245, 238, 250, 242]) if case == 'The networks agree' else np.array([180, 260, 210, 300, 150])
            u1, u2 = st.columns(2)
            u1.metric('Forecast (average of the 5)', f'{vals.mean():.0f}')
            u2.metric('Uncertainty (spread)', f'±{vals.std():.1f}')
            st.bar_chart(pd.DataFrame({'PM2.5 forecast': vals}, index=[f'Network {i}' for i in range(1, 6)]), height=220)
            points('On real validation days, when the networks agreed the forecast was off by 17.2 on average.',
                   'When they disagreed it was off by 27.4, about 59% more.',
                   'So the spread is a real warning, but a modest one: the networks also disagree more on dirty days.')

        with t6:
            points('The three methods were scored once on the test days, with all settings already fixed.',
                   'PM2.5 error, lower is better.')
            st.dataframe(pd.DataFrame({
                'Test period': ['Before the lockdown (288 days)', 'All 387 test days', 'Worst 10% of days'],
                'Tomorrow = today': [27.8, 23.8, 64.0], 'Linear regression': [26.2, 22.9, 62.1], '5-network ensemble': [26.7, 24.8, 60.2]}),
                hide_index=True, width='stretch')
            points('Before the lockdown, the ensemble is about 4% better than "tomorrow = today" and about 2% behind linear regression.',
                   'On the worst 10% of days it is the best of the three, and those days matter most.',
                   'During the lockdown it is clearly worse: the networks had never seen air that clean and over-predicted.')

    # ------------------------------------------------------------------ stage 2
    elif stage.startswith('2'):
        fz = load_fuzzy()
        st.subheader('Stage 2: turn the forecast into a risk class')
        st.write('The fuzzy system takes six inputs (forecast PM2.5, forecast PM10, forecast uncertainty, wind, humidity and season) '
                 'and gives a risk class (Low, Moderate or Severe), a 0 to 100 risk score, and the rules that fired.')
        f1, f2, f3 = st.tabs(['1. Fuzzify', '2. The 16 rules', '3. From rules to a score'])

        with f1:
            points('A hard cutoff calls 109 "not high" and 111 "high". Fuzzy sets let a value be partly in two groups at once.',
                   'Low and High use a trapezoid (a flat top of full membership). Medium uses a triangle (one peak at 1).')
            val = st.slider('Try a forecast PM2.5 value (µg/m³)', 0, 400, 130, key='how_slider')
            deg = {lab: float(_mf(k, p, np.array([val]))[0]) for lab, (k, p) in fz.variables['pm25'].items()}
            d1, d2, d3 = st.columns(3)
            d1.metric('Low', f'{deg["low"]:.2f}')
            d2.metric('Medium', f'{deg["medium"]:.2f}')
            d3.metric('High', f'{deg["high"]:.2f}')
            points('Slide the value and watch the three shares change.')
            st.dataframe(pd.DataFrame({
                'Input': ['PM2.5', 'PM10'], 'Low': ['Full up to 40, zero at 70', 'Full up to 70, zero at 110'],
                'Medium (start, peak, end)': ['50, 90, 140', '90, 180, 280'],
                'High': ['Starts at 110, full from 160', 'Starts at 240, full from 340']}),
                hide_index=True, width='stretch')
            st.markdown('**See every input\'s curves**')
            var = st.selectbox('Input', list(NAMES), format_func=lambda v: NAMES[v].capitalize(), key='how_var')
            if var == 'season':
                curve = pd.DataFrame({lab: p for lab, (_, p) in fz.variables['season'].items()}, index=range(1, 13))
                curve.index.name = 'month'
            else:
                hi = {'pm25': 400, 'pm10': 600}.get(var)
                if hi is None:
                    vals_ = [x for _, (_, p) in fz.variables[var].items() for x in p if abs(x) < 1e5]
                    lo_, hi = min(vals_) * 0.7, max(vals_) * 1.4
                else:
                    lo_ = 0
                xs_ = np.linspace(lo_, hi, 300)
                curve = pd.DataFrame({lab: _mf(k, p, xs_) for lab, (k, p) in fz.variables[var].items()}, index=np.round(xs_, 1))
                curve.index.name = NAMES[var]
            st.line_chart(curve, height=260)
            points('Uncertainty, wind and humidity cutoffs come from the forecast days used to design the rules '
                   '(uncertainty at the 40th and 60th percentiles, wind and humidity at quartiles).')

        with f2:
            points('16 IF-THEN rules written by hand from domain knowledge: 7 for Severe, 7 for Moderate and 2 for Low.',
                   'If no rule fires, the system falls back to Moderate.')
            order = {2: 'Severe', 1: 'Moderate', 0: 'Low'}
            rows_ = []
            for n, (a1, c1) in enumerate(HAND_RULES, 1):
                rows_.append({'No.': n, 'Rule': rule_plain(a1, c1), 'Class': CLASS_NAMES[c1]})
            st.dataframe(pd.DataFrame(rows_), hide_index=True, width='stretch')
            st.markdown('**The reasoning behind them**\n'
                        '- High pollution, calm wind and winter: pollutants stay trapped near the ground.\n'
                        '- Strong wind in the warmer or monsoon season: pollutants disperse more easily.\n'
                        '- Humid winter conditions: fog can add to pollutant build-up.\n'
                        '- High forecast uncertainty: the system takes a cautious line instead of confidently declaring Low.')

        with f3:
            points('The live demo uses Mamdani inference with centroid defuzzification.',
                   'It turns the rules that fired into one risk score from 0 to 100.')
            stages_ = [('1. Fuzzify', 'Convert inputs into membership degrees'), ('2. Rule evaluation', 'Rule strength is its weakest condition (MIN)'),
                       ('3. Aggregation', 'Combine the activated output shapes using MAX'), ('4. Defuzzify', 'Take the centroid to get one risk score')]
            st.markdown('<div class="flow">' + ''.join(f'<div class="box on"><b>{t}</b>{d}</div>' for t, d in stages_) + '</div>',
                        unsafe_allow_html=True)

            st.markdown('**Try it: set a day and watch the four stages**')
            g1, g2, g3 = st.columns(3)
            x25 = g1.slider('Forecast PM2.5', 0, 400, 130, key='mm_pm25')
            x10 = g2.slider('Forecast PM10', 0, 600, 200, key='mm_pm10')
            xu = g3.slider('Forecast spread', 0.0, 25.0, 6.0, 0.5, key='mm_unc')
            g4, g5, g6 = st.columns(3)
            xw = g4.slider('Wind (km/h)', 0.0, 20.0, 8.0, 0.5, key='mm_wind')
            xh = g5.slider('Humidity (%)', 0, 100, 60, key='mm_hum')
            MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            xm = g6.selectbox('Month', list(range(1, 13)), index=11, format_func=lambda m: MON[m - 1], key='mm_month')

            dd = {'pm25': [x25], 'pm10': [x10], 'unc': [xu], 'wind': [xw], 'humidity': [xh], 'season': [xm]}
            F = fz.firing(fz.fuzzify(dd))[0]                       # stages 1 and 2
            S = np.zeros(3)
            for j, (_, c1) in enumerate(fz.rules):
                S[c1] = max(S[c1], F[j])
            z = np.linspace(0, 100, 201)
            sets_ = {c1: _mf(k, p, z) for c1, (k, p) in OUT_SETS.items()}
            clip = {c1: np.minimum(S[c1], sets_[c1]) for c1 in range(3)}
            agg = np.maximum.reduce(list(clip.values()))           # stage 3
            score_ = float(fz.mamdani_score(dd)[0])                # stage 4
            band_ = int(fz.score_to_class(score_))

            lc, rc = st.columns([5, 6], gap='large')
            with lc:
                st.markdown('**Stage 2: rules that fired**')
                live = sorted([(F[j], j) for j in range(len(fz.rules)) if F[j] > 0], reverse=True)[:5]
                if not live:
                    st.markdown('No rule fired, so the score defaults to 50 (Moderate).')
                for strength, j in live:
                    ant_, c1 = fz.rules[j]
                    st.markdown(f'<div class="fired"><div>{strength:.2f}</div><div>{rule_plain(ant_, c1)}'
                                f'<div class="bar"><i style="width:{strength * 100:.0f}%;background:{COLORS[c1]}"></i></div></div></div>',
                                unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                k1.metric('Low', f'{S[0]:.2f}'); k2.metric('Moderate', f'{S[1]:.2f}'); k3.metric('Severe', f'{S[2]:.2f}')
                st.caption('The strongest rule of each class sets how high that class\'s output shape is allowed to rise.')
            with rc:
                st.markdown('**Stages 3 and 4: combine, then take the centroid**')
                df_agg = pd.DataFrame({'score': z, 'membership': agg})
                df_sets = pd.concat([pd.DataFrame({'score': z, 'membership': clip[c1], 'class': CLASS_NAMES[c1]}) for c1 in range(3)])
                xs = alt.X('score:Q', scale=alt.Scale(domain=[0, 100]), title='risk score')
                ys_ = alt.Y('membership:Q', scale=alt.Scale(domain=[0, 1]), title='degree')
                area = alt.Chart(df_agg).mark_area(opacity=.35, color='#2B4C63').encode(x=xs, y=ys_)
                lines = alt.Chart(df_sets).mark_line(strokeDash=[4, 3]).encode(
                    x=xs, y=ys_, color=alt.Color('class:N', legend=alt.Legend(title=None, orient='bottom'),
                                                 scale=alt.Scale(domain=CLASS_NAMES, range=[COLORS[0], COLORS[1], COLORS[2]])))
                rule_ = alt.Chart(pd.DataFrame({'score': [score_]})).mark_rule(color='#E4572E', strokeWidth=3).encode(x=xs)
                st.altair_chart(area + lines + rule_, height=230, width='stretch')
                st.caption('The shaded area is the combined shape (MAX). The orange line is its balance point, the centroid.')

            st.markdown(f'<div class="result" style="padding:1rem 1.4rem"><div class="label">Risk score {score_:.0f} out of 100, '
                        f'so the band is <b style="color:{COLORS[band_]}">{CLASS_NAMES[band_]}</b></div>{band_html(score_)}</div>',
                        unsafe_allow_html=True)

            st.markdown('')
            o1, o2 = st.columns(2, gap='large')
            with o1:
                st.markdown('**Output membership functions**')
                st.dataframe(pd.DataFrame({'Output class': ['Low', 'Moderate', 'Severe'],
                                           'Shape on the 0 to 100 score': ['Full from 0 to 15, falls to 0 at 40',
                                                                           'Triangle: 25, 50, 75', 'Starts at 60, full from 85 to 100']}),
                             hide_index=True, width='stretch')
            with o2:
                st.markdown('**Centroid formula**')
                st.latex(r'Z^{*}=\frac{\int z\,\mu(z)\,dz}{\int \mu(z)\,dz}')
                st.caption('Z* is the final risk score and μ(z) is the combined output shape. '
                           f'Bands: 0 to {SCORE_CUTS[0]:.0f} Low, {SCORE_CUTS[0]:.0f} to {SCORE_CUTS[1]:.0f} Moderate, above {SCORE_CUTS[1]:.0f} Severe.')
            st.markdown('<p class="note">On the Forecast tab, the headline class is the strongest rule vote and the score comes from this '
                        'centroid. Close to a band boundary the two can differ by one step.</p>', unsafe_allow_html=True)

    # ------------------------------------------------------------------ stage 3
    else:
        st.subheader('Stage 3: putting it together')
        st.write('Behind the scenes, the Forecast tab does these steps in order:')
        st.markdown('1. Take the last 3 days of readings and the month.\n'
                    '2. Scale them, and run all 5 networks.\n'
                    '3. Average the forecasts, and measure how far apart they are.\n'
                    '4. Pass forecast PM2.5, forecast PM10, the spread, wind, humidity and season to the fuzzy rules.\n'
                    '5. Show the risk class, the 0 to 100 score, and the rules that fired.')
        st.markdown('**Limits to keep in mind**')
        st.markdown(
            '- Yesterday already predicts today well (correlation 0.87), so any forecaster has little room to improve.\n'
            '- With about 1,100 training days, a neural network has little advantage over a plain linear model.\n'
            '- The networks over-predict when air is cleaner than anything they trained on, as in the 2020 lockdown.\n'
            '- The uncertainty is a modest reliability signal and partly tracks the pollution level.\n'
            '- The weather inputs come from a model reconstruction, not a ground station.\n'
            '- The risk class is a proxy taken from the official air-quality category, not a measured health outcome.')
        st.markdown('One line to take away: the system beats "tomorrow equals today" on forecasts, ties a linear model on normal days, '
                    'does best on the worst smog days, and explains every answer. It weakens on conditions far from anything it was trained on.')

# ============================================================================= Results
with tab_res:
    st.header('How well does it work?')
    st.subheader('Pollution forecast (validation period, error in µg/m³, lower is better)')
    st.dataframe(pd.DataFrame({
        'Method': ['Tomorrow equals today', 'Linear regression', '5-network ensemble'],
        'PM2.5 error': [25.2, 22.2, 22.3], 'PM10 error': [46.5, 40.0, 41.7],
        'PM2.5 R²': [0.72, 0.80, 0.80], 'PM10 R²': [0.70, 0.77, 0.75]}), hide_index=True, width='stretch')
    st.markdown('<p class="note">The ensemble beats "tomorrow equals today", ties linear regression on normal days and does best '
                'on the worst 10% of smog days (PM2.5 error 60.2 against 64.0 for persistence on the test period).</p>',
                unsafe_allow_html=True)

    st.subheader('Risk class (Low, Moderate, Severe)')
    period = st.radio('Scored on', ['Days used to design the rules', 'Test period (all days)', 'Test period before the lockdown'],
                      horizontal=True)
    T = {
        'Days used to design the rules': [
            ('Tomorrow\'s class = today\'s class', 'n/a', .840, .794, .686, .869, .826),
            ('Simple CPCB cutoffs', 'n/a', .866, .848, .857, .844, .913),
            ('Hand-written fuzzy rules (final)', '16', .842, .821, .857, .827, .870),
            ('Rules written automatically from data', '91', .912, .881, .914, .911, .913)],
        'Test period (all days)': [
            ('Tomorrow\'s class = today\'s class', 'n/a', .819, .789, .706, .855, .808),
            ('Simple CPCB cutoffs', 'n/a', .770, .726, .485, .784, .974),
            ('Hand-written fuzzy rules (final)', '16', .767, .713, .426, .797, .974),
            ('Rules written automatically from data', '91', .767, .698, .382, .855, .833)],
        'Test period before the lockdown': [
            ('Tomorrow\'s class = today\'s class', 'n/a', .806, .783, .711, .830, .808),
            ('Simple CPCB cutoffs', 'n/a', .753, .731, .578, .697, .974),
            ('Hand-written fuzzy rules (final)', '16', .753, .722, .533, .709, .974),
            ('Rules written automatically from data', '91', .760, .717, .511, .794, .833)]}[period]
    st.dataframe(pd.DataFrame(T, columns=['Method', 'Rules', 'Accuracy', 'Macro-F1', 'Recall Low', 'Recall Moderate', 'Recall Severe']),
                 hide_index=True, width='stretch')
    st.markdown('<p class="note">Macro-F1 gives Low, Moderate and Severe equal weight, which matters because Low days are rare. '
                'Scores on the days used to design the rules are flattering. The test period is the fair check, '
                'and there the simple methods are about as good as the fuzzy system.</p>', unsafe_allow_html=True)

    st.subheader('Could a genetic algorithm write better rules?')
    st.write('We let a genetic algorithm (GA) search for rules and cutoffs automatically, then scored it only on days it had '
             'never seen: the 387 design days were split into four blocks, the GA tuned on three and was scored on the fourth.')
    st.dataframe(pd.DataFrame({
        'Method': ['Simple CPCB cutoffs', 'Hand-written fuzzy rules', 'GA finds rules and cutoffs', 'GA tunes cutoffs only'],
        'Rules': ['0', '16', '2', '16'], 'Macro-F1 on unseen days': [0.848, 0.821, 0.788, 0.787]}),
        hide_index=True, width='stretch')
    st.markdown(
        '**Result: the GA did not beat the hand-written rules.** Its scores sit within about ±0.05 of the others, which is the '
        'size of the noise with so few days. Tuned on the days it was scored on, it reached 0.915, and that fell to 0.787 on '
        'unseen days, which is overfitting. The reasons:\n'
        '- Only 387 design days, and just 35 of them Low, is too little for a search to find rules that hold up.\n'
        '- The class is built from pollution levels, so a few simple rules capture almost all of it.\n'
        '- Results changed noticeably from one random start to the next (0.76 to 0.83).\n'
        '- The hand-written rules carry domain knowledge that small data cannot teach.\n\n'
        'What the GA did give us: a 2-rule system about as accurate as the 16 hand-written rules, which is much easier to read.')
    gp = HERE / 'results' / 'comparison_with_test.csv'
    if gp.exists():
        st.subheader('Final test-period comparison including the GA')
        st.dataframe(pd.read_csv(gp), hide_index=True, width='stretch')