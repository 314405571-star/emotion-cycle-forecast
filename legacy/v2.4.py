# coding:utf-8
"""
情绪周期监控模块 - 陈小群理论版
版本：v2.4
修复：明日预判依赖历史数据积累，测试用例改为单实例连续运行
新增：预判返回主预判+备选预判，提高实用性
适用于聚宽平台
"""

import numpy as np
from collections import deque
import datetime
import json
import pandas as pd
from jqdata import *
from jqlib.technical_analysis import *


class EmotionCycle:
    """情绪周期监控器 - 陈小群理论版（含明日预判）"""

    def __init__(self, config=None):
        """初始化情绪周期监控器"""
        self.config = {
            'thresholds': {
                'ice_abs': 30,
                'climax_abs': 75,
            },
            'sector_weights': {
                'main': 0.50,
                'growth': 0.25,
                'star': 0.15,
                'north': 0.10,
            },
            'indicator_weights': {
                'limit_ratio': 0.35,
                'up_ratio': 0.25,
                'seal_rate': 0.25,
                'big_loss_score': 0.15,
            },
            'position_ratios': {
                'ice': 0.20,
                'repair': 0.60,
                'rise': 0.80,
                'climax': 0.40,
                'divergence': 0.30,
                'fall': 0.10,
                'normal': 0.50,
            },
            'early_signals': {
                'climax_reduce_pct': 0.50,
                'ice_add_pct': 0.30,
                'climax_conditions': 3,
                'ice_conditions': 3,
            },
            'sampling': {
                'daily_count': 800,
                'intraday_count': 400,
                'min_stocks': 100,
            },
            'time_confidence': {
                '9:30': 0.30, '10:00': 0.55, '10:30': 0.70,
                '11:30': 0.80, '13:30': 0.85, '14:30': 0.95,
                '15:00': 0.98,
            },
            'max_daily_change': 15,
            'verbose': True,
        }
        if config:
            self._merge_config(config)
        self._init_storage()
        if self.config['verbose']:
            print("[情绪周期] 模块初始化完成 v2.4 (含明日预判)")

    def _merge_config(self, user_config):
        for key, value in user_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    def _init_storage(self):
        self.yesterday = {
            'emotion': 50.0,
            'phase': 'normal',
            'position_ratio': 0.5,
            'timestamp': None,
            'confidence': 0.98,
            'raw_data': None,
        }
        self.today = {}
        self.daily_emotions = deque(maxlen=30)
        self.daily_phases = deque(maxlen=30)
        self.weekly_emotions = deque(maxlen=12)
        self.volume_history = deque(maxlen=30)
        self.diffusion_history = deque(maxlen=10)
        self.current_phase = "normal"
        self.current_emotion = 50.0
        self.current_position_ratio = 0.5
        self.current_early_signal = None
        self.current_turning_points = []
        self.last_calculated_time = None
        self.yesterday_raw_data = None
        self._last_raw_data = None

    def _get_time_confidence(self, dt):
        time_key = dt.strftime('%H:%M')
        if time_key in self.config['time_confidence']:
            return self.config['time_confidence'][time_key]
        hour = dt.hour
        minute = dt.minute
        if hour == 9:
            return 0.30
        elif hour == 10:
            return 0.55 if minute < 30 else 0.70
        elif hour == 11:
            return 0.80
        elif hour == 13:
            return 0.80 if minute < 30 else 0.85
        elif hour == 14:
            return 0.90 if minute < 30 else 0.95
        else:
            return 0.50

    # ==================== 核心情绪判定 ====================
    def _determine_phase(self, emotion, diffusion, raw_data):
        prev_phase = self.yesterday.get('phase', 'normal')
        prev_emotion = self.yesterday.get('emotion', 50.0)
        if not raw_data:
            return prev_phase
        limit_ratio = raw_data.get('limit_ratio', 0)
        up_ratio = raw_data.get('up_ratio', 0.5)
        seal_rate = raw_data.get('seal_rate', 0.5)
        big_loss_score = raw_data.get('big_loss_score', 0.5)

        if limit_ratio > 0.04 and seal_rate > 0.70 and up_ratio > 0.60:
            return "climax"
        if up_ratio < 0.25 and big_loss_score < 0.30:
            return "ice"
        if prev_phase == "climax" and emotion < prev_emotion - 5:
            return "divergence"
        if prev_phase in ("ice", "divergence", "fall") and emotion > prev_emotion + 5:
            if up_ratio > 0.40:
                return "repair"
        if emotion < prev_emotion - 3 and emotion > 30:
            return "fall"
        if emotion > prev_emotion + 3 and prev_phase not in ("climax", "ice"):
            return "rise"
        return prev_phase

    # ==================== 转折检测 ====================
    def _detect_turning_points(self, raw_data, current_emotion, phase):
        signals = []
        prev_phase = self.yesterday.get('phase', 'normal')
        prev_emotion = self.yesterday.get('emotion', 50.0)
        yesterday_raw = self.yesterday_raw_data
        if yesterday_raw is None:
            return signals
        if prev_phase == "divergence" and phase in ("repair", "rise", "climax"):
            limit_ratio = raw_data.get('limit_ratio', 0)
            seal_rate = raw_data.get('seal_rate', 0.5)
            prev_limit = yesterday_raw.get('limit_ratio', 0)
            prev_seal = yesterday_raw.get('seal_rate', 0.5)
            if (limit_ratio > prev_limit * 1.2 and seal_rate > prev_seal * 1.1):
                signals.append({
                    'type': 'turn_strong',
                    'desc': '分歧转一致（买入信号）',
                    'urgency': 1
                })
        if prev_phase == "ice" and phase == "repair":
            up_ratio = raw_data.get('up_ratio', 0.5)
            if up_ratio > 0.40:
                signals.append({
                    'type': 'turn_repair',
                    'desc': '冰点转修复（左侧布局信号）',
                    'urgency': 1
                })
        return signals

    # ==================== 明日周期预判（增强版） ====================
    def predict_next_phase(self, num_days=7):
        """
        基于最近 num_days 个交易日的情绪数据，预判下一个交易日的周期阶段。
        返回: (主预判, 备选预判, 置信度, 描述, 观察要点)
        """
        if len(self.daily_emotions) < 3:
            return None, None, 0, "历史数据不足，无法预判", ""

        recent_emotions = list(self.daily_emotions)[-num_days:]
        recent_phases = list(self.daily_phases)[-num_days:]
        current_phase = recent_phases[-1]
        current_emotion = recent_emotions[-1]

        # 计算斜率（每日变化量）
        if len(recent_emotions) >= 2:
            x = np.arange(len(recent_emotions))
            slope = np.polyfit(x, recent_emotions, 1)[0]
        else:
            slope = 0

        # 计算近两日变化量
        if len(recent_emotions) >= 2:
            last_change = recent_emotions[-1] - recent_emotions[-2]
        else:
            last_change = 0

        # 初始化
        main_pred = current_phase
        alt_pred = None
        confidence = 0.5
        desc = ""
        watch = ""

        # 根据当前阶段和斜率、最近变化进行预测
        if current_phase == "ice":
            if slope > 2 or last_change > 5:
                main_pred = "repair"
                confidence = 0.7
                desc = "冰点后情绪快速回暖，大概率进入修复期"
                watch = "关注竞价高开和量能放大"
                alt_pred = "ice"
            elif slope > 0:
                main_pred = "repair"
                confidence = 0.55
                desc = "冰点后情绪缓慢回升，可能进入弱修复"
                watch = "观察开盘后能否站稳分时均线"
                alt_pred = "ice"
            else:
                main_pred = "ice"
                confidence = 0.6
                desc = "冰点后情绪继续低迷，可能延续冰点"
                watch = "等待放量杀跌后的企稳信号"
                alt_pred = "repair"

        elif current_phase == "repair":
            if slope > 3 or last_change > 8:
                main_pred = "rise"
                confidence = 0.7
                desc = "修复期情绪加速上升，有望进入上升期"
                watch = "关注涨停家数和封板率"
                alt_pred = "repair"
            elif slope > 0:
                main_pred = "repair"
                confidence = 0.6
                desc = "修复期情绪温和回升，可能持续修复"
                watch = "观察上涨比是否持续改善"
                alt_pred = "divergence"
            else:
                main_pred = "divergence"
                confidence = 0.55
                desc = "修复期情绪出现回落，可能转为分歧"
                watch = "注意前期强势股是否出现亏钱效应"
                alt_pred = "repair"

        elif current_phase == "rise":
            if slope > 5 or last_change > 10:
                main_pred = "climax"
                confidence = 0.7
                desc = "上升期情绪加速，大概率进入高潮期"
                watch = "高潮期需警惕，逐步止盈"
                alt_pred = "rise"
            elif slope > 0:
                main_pred = "rise"
                confidence = 0.65
                desc = "上升期情绪平稳上升，可能持续"
                watch = "持股为主，观察是否出现高潮信号"
                alt_pred = "climax"
            else:
                main_pred = "divergence"
                confidence = 0.6
                desc = "上升期情绪出现滞涨，可能转入分歧"
                watch = "减仓观望，等待方向选择"
                alt_pred = "rise"

        elif current_phase == "climax":
            if slope < -3 or last_change < -5:
                main_pred = "divergence"
                confidence = 0.7
                desc = "高潮后情绪快速回落，大概率进入分歧期"
                watch = "竞价若低开且量能萎缩，则确认分歧"
                alt_pred = "fall"
            elif slope < 0:
                main_pred = "divergence"
                confidence = 0.6
                desc = "高潮后情绪缓慢回落，可能进入分歧"
                watch = "关注龙头股表现，若走弱则减仓"
                alt_pred = "climax"
            else:
                main_pred = "climax"
                confidence = 0.5
                desc = "高潮期情绪仍强，可能延续，但风险加大"
                watch = "高潮延续需放量，否则为强弩之末"
                alt_pred = "divergence"

        elif current_phase == "divergence":
            if slope > 2 or last_change > 5:
                main_pred = "repair"
                confidence = 0.7
                desc = "分歧后情绪回升，可能转为修复"
                watch = "关注竞价爆量高开，确认转一致"
                alt_pred = "rise"
            elif slope > 0:
                main_pred = "repair"
                confidence = 0.55
                desc = "分歧后情绪小幅回暖，可能弱修复"
                watch = "观察封板率是否改善"
                alt_pred = "divergence"
            else:
                main_pred = "fall"
                confidence = 0.65
                desc = "分歧后情绪继续下行，大概率进入退潮期"
                watch = "空仓回避，等待冰点"
                alt_pred = "ice"

        elif current_phase == "fall":
            if slope > 1 or last_change > 2:
                main_pred = "ice"
                confidence = 0.6
                desc = "退潮期情绪下跌趋缓，可能进入冰点"
                watch = "冰点后可左侧布局，但需等待明确信号"
                alt_pred = "repair"
            else:
                main_pred = "fall"
                confidence = 0.6
                desc = "退潮期延续，继续下跌"
                watch = "严控仓位，不轻易抄底"
                alt_pred = "ice"

        else:  # normal
            if current_emotion > 55:
                main_pred = "rise"
                alt_pred = "repair"
            elif current_emotion < 45:
                main_pred = "repair"
                alt_pred = "ice"
            else:
                main_pred = "normal"
                alt_pred = "repair"
            confidence = 0.5
            desc = "正常期，方向不明，需结合竞价判断"
            watch = "观察开盘后30分钟情绪走向"

        return main_pred, alt_pred, confidence, desc, watch

    # ==================== 原有方法（未变） ====================
    def update(self, context, current_data, intraday=False):
        current_time = datetime.datetime.now()
        confidence = self._get_time_confidence(current_time)
        raw_data = self._fetch_market_data(context, current_data, end_date=None)
        if raw_data is None:
            return self._get_fallback_state()
        raw_emotion = self._calculate_raw_emotion(raw_data)
        if confidence < 0.60:
            final_emotion = self.yesterday['emotion'] * (1 - confidence) + raw_emotion * confidence
        else:
            final_emotion = raw_emotion
        change = final_emotion - self.yesterday['emotion']
        if abs(change) > self.config['max_daily_change']:
            final_emotion = self.yesterday['emotion'] + self.config['max_daily_change'] * (1 if change > 0 else -1)
        diffusion = self._calculate_diffusion(raw_data)
        phase = self._determine_phase(final_emotion, diffusion, raw_data)
        position_ratio = self.config['position_ratios'].get(phase, 0.5)
        early_signal = None
        if confidence >= 0.70:
            early_signal = self._detect_early_signal(final_emotion, diffusion, phase)
        turning_points = []
        if confidence >= 0.70:
            turning_points = self._detect_turning_points(raw_data, final_emotion, phase)
        self.current_emotion = final_emotion
        self.current_phase = phase
        self.current_position_ratio = position_ratio
        self.current_early_signal = early_signal
        self.current_turning_points = turning_points
        self.last_calculated_time = current_time
        if self.config['verbose']:
            print("[情绪周期] {:.0%}置信 | 情绪:{:.1f} | {} | 仓位:{:.0%}".format(
                confidence, final_emotion, phase, position_ratio))
            if turning_points:
                for tp in turning_points:
                    print("  [转折] {}".format(tp['desc']))
        return phase, final_emotion, position_ratio, early_signal

    def _fetch_market_data(self, context, current_data, end_date=None):
        try:
            all_stocks = list(get_all_securities(['stock']).index)
            if not all_stocks:
                return None
            sample_size = min(self.config['sampling']['daily_count'], len(all_stocks))
            import random
            seed = int(end_date.replace('-', '')) if end_date else 42
            random.seed(seed)
            stocks = random.sample(all_stocks, sample_size)
            total_data = []
            if current_data:
                for stock in stocks:
                    if stock not in current_data:
                        continue
                    cd = current_data[stock]
                    last = cd.last_price if cd.last_price > 0 else cd.close
                    prev = cd.pre_close
                    up_limit = cd.high_limit
                    if last <= 0 or prev <= 0:
                        continue
                    change = (last - prev) / prev
                    total_data.append({
                        'code': stock, 'last': last, 'prev': prev,
                        'change': change, 'up_limit': up_limit,
                        'is_up': last > prev,
                        'is_limit_up': up_limit > 0 and last >= up_limit - 0.01,
                        'hit_limit': up_limit > 0 and last >= up_limit - 0.01,
                        'is_big_loss': change <= -0.07,
                    })
            else:
                price_df = get_price(
                    stocks, end_date=end_date, frequency='1d',
                    fields=['close', 'high', 'high_limit'],
                    count=2, skip_paused=False, panel=False
                )
                if price_df is None or price_df.empty:
                    return None
                for stock, grp in price_df.groupby('code'):
                    grp = grp.sort_values('time')
                    if len(grp) < 2:
                        continue
                    last = grp['close'].iloc[-1]
                    prev = grp['close'].iloc[-2]
                    up_limit = grp['high_limit'].iloc[-1] if 'high_limit' in grp.columns else last * 1.1
                    high = grp['high'].iloc[-1] if 'high' in grp.columns else last
                    if last <= 0 or prev <= 0:
                        continue
                    change = (last - prev) / prev
                    total_data.append({
                        'code': stock, 'last': last, 'prev': prev,
                        'change': change, 'up_limit': up_limit,
                        'is_up': last > prev,
                        'is_limit_up': up_limit > 0 and last >= up_limit - 0.01,
                        'hit_limit': up_limit > 0 and high >= up_limit - 0.01,
                        'is_big_loss': change <= -0.07,
                    })
            total = len(total_data)
            if total < self.config['sampling']['min_stocks']:
                return None
            up_count = sum(1 for d in total_data if d['is_up'])
            limit_up_count = sum(1 for d in total_data if d['is_limit_up'])
            hit_limit_count = sum(1 for d in total_data if d['hit_limit'])
            big_loss_count = sum(1 for d in total_data if d['is_big_loss'])
            seal_rate = limit_up_count / hit_limit_count if hit_limit_count > 0 else 0.5
            strong_count = sum(1 for d in total_data if d['change'] > 0.05)
            mid_count = sum(1 for d in total_data if 0.02 < d['change'] <= 0.05)
            return {
                'up_ratio': up_count / total,
                'limit_ratio': limit_up_count / total,
                'seal_rate': seal_rate,
                'big_loss_score': max(0, 1 - (big_loss_count / 40)),
                'strong_count': strong_count,
                'mid_count': mid_count,
                'total_count': total,
            }
        except Exception as e:
            print("数据获取失败: {}".format(e))
            return None

    def _calculate_raw_emotion(self, raw_data):
        if not raw_data:
            return 50
        weights = self.config['indicator_weights']
        limit_score = min(100.0, raw_data['limit_ratio'] * 1000.0)
        up_score = min(100.0, max(0.0, (raw_data['up_ratio'] - 0.30) / 0.40 * 100.0))
        seal_score = raw_data['seal_rate'] * 100.0
        loss_score = raw_data['big_loss_score'] * 100.0
        emotion = (
            limit_score * weights['limit_ratio'] +
            up_score    * weights['up_ratio'] +
            seal_score  * weights['seal_rate'] +
            loss_score  * weights['big_loss_score']
        )
        return min(100.0, max(0.0, emotion))

    def _calculate_diffusion(self, raw_data):
        if not raw_data:
            return 50
        strong = raw_data.get('strong_count', 0)
        mid = raw_data.get('mid_count', 0)
        if strong == 0:
            return 50
        diffusion = (mid / strong) * 100
        self.diffusion_history.append(diffusion)
        return min(100, diffusion)

    def _detect_early_signal(self, emotion, diffusion, phase):
        if phase == 'climax' and diffusion < 35:
            return {'type': 'climax', 'urgency': 2}
        elif phase == 'ice' and diffusion > 55:
            return {'type': 'ice', 'urgency': 2}
        return None

    def _get_fallback_state(self):
        return (self.yesterday['phase'], self.yesterday['emotion'],
                self.yesterday['position_ratio'], None)

    def save_daily_state(self, context):
        self.yesterday = {
            'emotion': self.current_emotion,
            'phase': self.current_phase,
            'position_ratio': self.current_position_ratio,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': 0.98,
            'raw_data': self._last_raw_data,
        }
        self.daily_emotions.append(self.current_emotion)
        self.daily_phases.append(self.current_phase)
        print("[情绪周期] 收盘保存 | 情绪:{:.1f} | 周期:{}".format(
            self.current_emotion, self.current_phase))
        return self.current_emotion, self.current_phase

    def get_persisted_state(self):
        return {
            'yesterday': self.yesterday,
            'daily_emotions': list(self.daily_emotions),
            'daily_phases': list(self.daily_phases),
        }

    def load_persisted_state(self, state):
        if 'yesterday' in state:
            self.yesterday = state['yesterday']
            self.current_emotion = self.yesterday['emotion']
            self.current_phase = self.yesterday['phase']
            self.current_position_ratio = self.yesterday['position_ratio']
        if 'daily_emotions' in state:
            self.daily_emotions = deque(state['daily_emotions'], maxlen=30)
        if 'daily_phases' in state:
            self.daily_phases = deque(state['daily_phases'], maxlen=30)
        print("[情绪周期] 加载历史状态成功")
        return True

    def get_position_ratio(self):
        return self.current_position_ratio

    def get_phase(self):
        return self.current_phase

    def get_emotion(self):
        return self.current_emotion

    def should_early_sell(self):
        if self.last_calculated_time:
            confidence = self._get_time_confidence(self.last_calculated_time)
            if confidence < 0.70:
                return False
        return (self.current_early_signal is not None and
                self.current_early_signal['type'] == 'climax')

    def should_early_buy(self):
        if self.last_calculated_time:
            confidence = self._get_time_confidence(self.last_calculated_time)
            if confidence < 0.70:
                return False
        return (self.current_early_signal is not None and
                self.current_early_signal['type'] == 'ice')

    def get_summary(self):
        confidence = 0
        if self.last_calculated_time:
            confidence = self._get_time_confidence(self.last_calculated_time)
        signal = self.current_early_signal['type'] if self.current_early_signal else '无'
        turning_desc = '; '.join([tp['desc'] for tp in self.current_turning_points]) if self.current_turning_points else '无'
        return """
        ==========================================
        情绪周期监控摘要 陈小群理论)
        ==========================================
        当前周期: {}
        情绪指数: {:.1f}
        建议仓位: {:.0%}
        置信度: {:.0%}
        提前信号: {}
        转折信号: {}
        ==========================================
        """.format(self.current_phase, self.current_emotion,
                   self.current_position_ratio, confidence, signal, turning_desc)

    # ==================== 研究环境专用方法 ====================
    def update_for_research(self, date_str):
        import datetime as _dt
        fake_time = _dt.datetime.strptime(date_str + ' 15:00:00', '%Y-%m-%d %H:%M:%S')
        self.last_calculated_time = fake_time
        confidence = 0.98
        raw_data = self._fetch_market_data(None, None, end_date=date_str)
        if raw_data is None:
            phase = self.yesterday['phase']
            emotion = self.yesterday['emotion']
            ratio = self.yesterday['position_ratio']
            signal = None
            turning = []
        else:
            self._last_raw_data = raw_data
            raw_emotion = self._calculate_raw_emotion(raw_data)
            final_emotion = self.yesterday['emotion'] * 0.02 + raw_emotion * 0.98
            change = final_emotion - self.yesterday['emotion']
            if abs(change) > self.config['max_daily_change']:
                final_emotion = self.yesterday['emotion'] + self.config['max_daily_change'] * (1 if change > 0 else -1)
            diffusion = self._calculate_diffusion(raw_data)
            phase = self._determine_phase(final_emotion, diffusion, raw_data)
            ratio = self.config['position_ratios'].get(phase, 0.5)
            signal = self._detect_early_signal(final_emotion, diffusion, phase)
            turning = self._detect_turning_points(raw_data, final_emotion, phase)
            emotion = final_emotion
            self.current_emotion = final_emotion
            self.current_phase = phase
            self.current_position_ratio = ratio
            self.current_early_signal = signal
            self.current_turning_points = turning
            self._last_raw_data = raw_data
        # 持久化到历史序列
        self.daily_emotions.append(self.current_emotion)
        self.daily_phases.append(self.current_phase)
        # 保存为昨日基准，供下一日使用
        self.yesterday = {
            'emotion': self.current_emotion,
            'phase': self.current_phase,
            'position_ratio': self.current_position_ratio,
            'timestamp': date_str + ' 15:00:00',
            'confidence': 0.98,
            'raw_data': self._last_raw_data,
        }
        self.yesterday_raw_data = self._last_raw_data
        return phase, emotion, ratio, signal


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    测试代码（单实例连续运行，修复预判）                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    import datetime as dt
    import pandas as pd
    from jqdata import get_trade_days

    today = dt.date.today()
    # 获取最近10个交易日，确保有足够历史
    recent_days = get_trade_days(end_date=today.strftime('%Y-%m-%d'), count=10)
    latest_trade_day = recent_days[-1]
    latest_str = latest_trade_day.strftime('%Y-%m-%d')

    print('\n' + '=' * 70)
    print(f'  情绪周期 v2.4 (陈小群理论 + 明日预判) 测试 —— 运行时间: {today.strftime("%Y-%m-%d %H:%M")}')
    print(f'  最新交易日: {latest_str}')
    print('=' * 70)

    # ---- 使用单个实例连续分析最近7天 ----
    print('\n' + '-' * 70)
    print('  第一部分：连续分析最近7个交易日（单实例）')
    print('-' * 70)

    # 取最近7个交易日用于分析
    lookback_days = get_trade_days(end_date=latest_str, count=7)
    print(f'  分析区间: {lookback_days[0].strftime("%Y-%m-%d")} ~ {lookback_days[-1].strftime("%Y-%m-%d")}  (共 {len(lookback_days)} 个交易日)')

    # 创建唯一实例
    ec = EmotionCycle()
    ec.config['verbose'] = False

    results = []
    for i, day in enumerate(lookback_days):
        day_str = day.strftime('%Y-%m-%d')
        print(f'  [{i+1}/{len(lookback_days)}] 分析 {day_str} ...', end='\r')
        try:
            ph, em, ra, sg = ec.update_for_research(day_str)
            sg_type = sg['type'] if sg else '无'
            tp = ec.current_turning_points
            tp_desc = '; '.join([t['desc'] for t in tp]) if tp else '无'
            results.append({
                '日期': day_str,
                '情绪指数': round(em, 1),
                '周期阶段': ph,
                '建议仓位': round(ra, 2),
                '提前信号': sg_type,
                '转折信号': tp_desc,
            })
        except Exception as e:
            print(f'\n  [警告] {day_str} 分析失败: {e}')

    print('\n')
    df = pd.DataFrame(results)

    if not df.empty:
        # 显示当前最新状态
        latest_row = df.iloc[-1]
        phase_label = {
            'ice': '🧊 冰点期', 'repair': '🔄 修复期', 'rise': '📈 上升期',
            'climax': '🔥 高潮期', 'divergence': '⚡ 分歧期', 'fall': '📉 退潮期',
            'normal': '⚖️ 正常期'
        }.get(latest_row['周期阶段'], latest_row['周期阶段'])

        phase_advice = {
            'ice': '轻仓观望，等待冰点转修复',
            'repair': '右侧加仓，参与修复行情',
            'rise': '满仓持股，让利润奔跑',
            'climax': '逐步止盈，高点兑现',
            'divergence': '减仓观望，等待分歧转一致',
            'fall': '空仓回避，等待冰点',
            'normal': '中性仓位，灵活应对'
        }.get(latest_row['周期阶段'], '')

        signal_label = latest_row['提前信号'] if latest_row['提前信号'] != '无' else '无'
        turning_desc = latest_row['转折信号'] if latest_row['转折信号'] != '无' else '无'

        print(f'\n  ┌──────────────────────────────────────────────────┐')
        print(f'  │  最新日期   : {latest_row["日期"]:>28s} │')
        print(f'  │  情绪指数   : {latest_row["情绪指数"]:>27.1f} │')
        print(f'  │  周期阶段   : {phase_label:>20s}             │')
        print(f'  │  阶段说明   : {phase_advice:>20s}             │')
        print(f'  │  建议仓位   : {latest_row["建议仓位"]:>27.0%} │')
        print(f'  │  提前信号   : {signal_label:>20s}             │')
        print(f'  │  转折信号   : {turning_desc:>20s}             │')
        print(f'  └──────────────────────────────────────────────────┘')

    # ---- 新增：明日周期预判（现在历史数据已积累） ----
    print('\n' + '-' * 70)
    print('  第二部分：明日周期预判（基于已积累的7日历史数据）')
    print('-' * 70)

    main_pred, alt_pred, pred_conf, pred_desc, pred_watch = ec.predict_next_phase(num_days=7)
    if main_pred:
        phase_label_full = {
            'ice': '🧊 冰点期', 'repair': '🔄 修复期', 'rise': '📈 上升期',
            'climax': '🔥 高潮期', 'divergence': '⚡ 分歧期', 'fall': '📉 退潮期',
            'normal': '⚖️ 正常期'
        }
        main_label = phase_label_full.get(main_pred, main_pred)
        alt_label = phase_label_full.get(alt_pred, alt_pred) if alt_pred else '无'
        print(f'  📅 主预判 : {main_label}  (置信度 {pred_conf:.0%})')
        print(f'  🔄 备选预判 : {alt_label}')
        print(f'  💬 描述 : {pred_desc}')
        print(f'  🔍 竞价观察要点 : {pred_watch}')
    else:
        print('  ⚠️ 历史数据不足，无法预判明日周期')

    # ---- 第三部分：显示最近7日汇总表 ----
    print('\n' + '-' * 70)
    print('  第三部分：最近 7 个交易日情绪周期汇总')
    print('-' * 70)

    if not df.empty:
        def classify_emotion(val):
            if val >= 75:
                return '🔥 高潮'
            elif val >= 60:
                return '📈 上升'
            elif val >= 45:
                return '⚖️ 正常'
            elif val >= 30:
                return '🔄 修复'
            else:
                return '🧊 冰点'

        print('  【最近 7 日情绪周期汇总】')
        print('  ' + '-' * 58)
        for idx, row in df.iterrows():
            marker = ' ← 最新' if row['日期'] == latest_str else ''
            cls = classify_emotion(row['情绪指数'])
            tp = row['转折信号']
            tp_str = f' 转:{tp}' if tp != '无' else ''
            print(f'  {row["日期"]}{marker}')
            print(f'    情绪={row["情绪指数"]:>5.1f}  {cls}  |  '
                  f'仓位={row["建议仓位"]:.0%}  |  '
                  f'阶段={row["周期阶段"]} {tp_str}')
        print('  ' + '-' * 58)

    # ---- 第四部分：趋势判断 ----
    print('\n' + '-' * 70)
    print('  第四部分：7日周期趋势判断')
    print('-' * 70)

    if not df.empty:
        emotions = df['情绪指数'].tolist()
        ratios = df['建议仓位'].tolist()
        phases = df['周期阶段'].tolist()

        emo_start, emo_end = emotions[0], emotions[-1]
        emo_delta = emo_end - emo_start
        if emo_delta >= 5:
            trend = '📈 情绪显著升温'
        elif emo_delta >= 2:
            trend = '📈 情绪小幅升温'
        elif emo_delta <= -5:
            trend = '📉 情绪显著降温'
        elif emo_delta <= -2:
            trend = '📉 情绪小幅降温'
        else:
            trend = '➡️ 情绪平稳'

        ratio_start, ratio_end = ratios[0], ratios[-1]
        ratio_direction = '递增' if ratio_end > ratio_start else ('递减' if ratio_end < ratio_start else '持平')

        from collections import Counter
        phase_counts = Counter(phases)
        phase_dist = ' → '.join([f'{p}x{c}' for p, c in phase_counts.most_common()])

        phase_label_lookup = {
            'ice': '🧊 冰点期', 'repair': '🔄 修复期', 'rise': '📈 上升期',
            'climax': '🔥 高潮期', 'divergence': '⚡ 分歧期', 'fall': '📉 退潮期',
            'normal': '⚖️ 正常期'
        }

        print(f'  情绪趋势  : {trend}  ({emo_start:.1f} → {emo_end:.1f}, Δ={emo_delta:+.1f})')
        print(f'  仓位趋势  : {ratio_start:.0%} → {ratio_end:.0%}  ({ratio_direction})')
        print(f'  周期分布  : {phase_dist}')
        print(f'  起点阶段  : {phase_label_lookup.get(phases[0], phases[0])}')
        print(f'  终点阶段  : {phase_label_lookup.get(phases[-1], phases[-1])}')

        print('\n  【综合判断（陈小群视角）】')
        current_phase = phases[-1]
        current_emotion_val = emotions[-1]

        if current_phase == 'ice':
            print('  🧊 冰点期 — 恐慌释放中，等待冰点转修复信号')
        elif current_phase == 'repair':
            print('  🔄 修复期 — 情绪回暖，可逐步加仓')
        elif current_phase == 'rise':
            print('  📈 上升期 — 赚钱效应好，满仓持股')
        elif current_phase == 'climax':
            print('  🔥 高潮期 — 情绪过热，注意逐步止盈')
        elif current_phase == 'divergence':
            print('  ⚡ 分歧期 — 分化加大，等待分歧转一致')
        elif current_phase == 'fall':
            print('  📉 退潮期 — 建议空仓观望')
        else:
            print('  ⚖️ 正常期 — 中性仓位灵活应对')

        if current_emotion_val >= 75:
            print('  ⚠️  情绪值进入高潮区间（≥75），注意高潮预警')
        elif current_emotion_val <= 30:
            print('  ℹ️  情绪值进入冰点区间（≤30），关注冰点转修复信号')

    # ---- 第五部分：历史分位数（可选） ----
    if not df.empty:
        try:
            lookback20 = get_trade_days(end_date=latest_str, count=21)
            lookback20_prev = lookback20[0: -len(lookback_days)]
            if len(lookback20_prev) > 0:
                print('\n' + '-' * 70)
                print('  第五部分：历史分位数参考（前14日）')
                print('-' * 70)
                # 单独实例获取历史数据，不污染当前实例
                ec_hist = EmotionCycle()
                ec_hist.config['verbose'] = False
                prev_results = []
                for day in lookback20_prev:
                    day_str = day.strftime('%Y-%m-%d')
                    try:
                        ph, em, ra, sg = ec_hist.update_for_research(day_str)
                        prev_results.append(em)
                    except Exception:
                        continue
                if prev_results:
                    import numpy as np
                    prev_arr = np.array(prev_results)
                    pct = (prev_arr < emotions[-1]).sum() / len(prev_arr) * 100
                    print(f'  前 {len(lookback20_prev)} 日情绪均值: {prev_arr.mean():.1f}')
                    print(f'  前 {len(lookback20_prev)} 日情绪范围: [{prev_arr.min():.1f}, {prev_arr.max():.1f}]')
                    print(f'  当前情绪 {emotions[-1]:.1f} 位于历史 {pct:.0f}% 分位')
        except Exception:
            pass

    print('\n' + '=' * 70)
    print('  测试完成！明日预判结果已输出，请结合集合竞价主力意图手工确认。')
    print('=' * 70)
    print()

    # 显示DataFrame
    df
收起代码 ↑
======================================================================
  情绪周期 v2.4 (野人哥理论 + 明日预判) 测试 —— 运行时间: 2026-07-09 00:00
  最新交易日: 2026-07-09
======================================================================
----------------------------------------------------------------------
  第一部分：连续分析最近7个交易日（单实例）
----------------------------------------------------------------------
  分析区间: 2026-07-01 ~ 2026-07-09  (共 7 个交易日)
[情绪周期] 模块初始化完成 v2.4 (含明日预判)
  [7/7] 分析 2026-07-09 ...


  ┌──────────────────────────────────────────────────┐
  │  最新日期   :                   2026-07-09 │
  │  情绪指数   :                        43.3 │
  │  周期阶段   :                🔄 修复期             │
  │  阶段说明   :          右侧加仓，参与修复行情             │
  │  建议仓位   :                         60% │
  │  提前信号   :                    无             │
  │  转折信号   :        冰点转修复（左侧布局信号）             │
  └──────────────────────────────────────────────────┘

----------------------------------------------------------------------
  第二部分：明日周期预判（基于已积累的7日历史数据）
----------------------------------------------------------------------
  📅 主预判 : 📈 上升期  (置信度 70%)
  🔄 备选预判 : 🔄 修复期
  💬 描述 : 修复期情绪加速上升，有望进入上升期
  🔍 竞价观察要点 : 关注涨停家数和封板率

----------------------------------------------------------------------
  第三部分：最近 7 个交易日情绪周期汇总
----------------------------------------------------------------------
  【最近 7 日情绪周期汇总】
  ----------------------------------------------------------
  2026-07-01
    情绪= 65.0  📈 上升  |  仓位=40%  |  阶段=climax
  2026-07-02
    情绪= 50.0  ⚖️ 正常  |  仓位=30%  |  阶段=divergence
  2026-07-03
    情绪= 59.0  ⚖️ 正常  |  仓位=60%  |  阶段=repair  转:分歧转一致（买入信号）
  2026-07-06
    情绪= 44.0  🔄 修复  |  仓位=10%  |  阶段=fall
  2026-07-07
    情绪= 29.0  🧊 冰点  |  仓位=20%  |  阶段=ice
  2026-07-08
    情绪= 28.3  🧊 冰点  |  仓位=20%  |  阶段=ice
  2026-07-09 ← 最新
    情绪= 43.3  🔄 修复  |  仓位=60%  |  阶段=repair  转:冰点转修复（左侧布局信号）
  ----------------------------------------------------------

----------------------------------------------------------------------
  第四部分：7日周期趋势判断
----------------------------------------------------------------------
  情绪趋势  : 📉 情绪显著降温  (65.0 → 43.3, Δ=-21.7)
  仓位趋势  : 40% → 60%  (递增)
  周期分布  : repairx2 → icex2 → climaxx1 → divergencex1 → fallx1
  起点阶段  : 🔥 高潮期
  终点阶段  : 🔄 修复期

  【综合判断（野人哥视角）】
  🔄 修复期 — 情绪回暖，可逐步加仓

----------------------------------------------------------------------
  第五部分：历史分位数参考（前14日）
----------------------------------------------------------------------
[情绪周期] 模块初始化完成 v2.4 (含明日预判)
  前 14 日情绪均值: 40.4
  前 14 日情绪范围: [22.1, 61.2]
  当前情绪 43.3 位于历史 57% 分位

======================================================================
  测试完成！明日预判结果已输出，请结合集合竞价主力意图手工确认。
======================================================================

收起输出 ↑
