import numpy as np
from pyqubo import Binary, Constraint
import openjij as oj
import time

def simulated_annealing(cannot_sleep, target_sleep_time, ideal_segments):
    cannot_sleep = np.array(cannot_sleep)  # list → NumPy配列に変換

    # 実行時間の計測開始
    total_start_time = time.time()

    # 1週間（7日間）、1日96スロット（15分刻み）
    num_days = 7
    num_slots = 96  # 1日 = 24時間 × 4（15分ごと）
    num_vars = num_days * num_slots

    #ペナルティの重み
    penalty_90min = 1  # 90分（6スロット）の倍数にするための制約
    continuous_sleep_weight = 10  # 連続睡眠のペナルティ重みを大きくする
    cannot_sleep_weight = 50  # 寝れない時間のペナルティ重みを大きくする
    segments_weight = 30  # 分割数のペナルティ重みを大きくする
    day_change_wight = 2  # 曜日間変動のペナルティ重みを大きくする
    target_sleep_wight = 20  # 目標睡眠時間のペナルティ重みを大きくする

    # QUBO変数の設定
    start_time_qubo_vars = time.time()
    x = {}
    for day in range(num_days):
        for slot in range(num_slots):
            x[(day, slot)] = Binary(f"x_{day}_{slot}")
    end_time_qubo_vars = time.time()
    print(f"QUBO Variables Setup Time: {end_time_qubo_vars - start_time_qubo_vars:.4f} seconds")


    # 目標睡眠時間との差分をペナルティとして追加
    start_time_target_sleep = time.time()
    penalty_target_sleep = 0
    for day in range(num_days):
        # 睡眠スロットの合計数
        sleep_sum = sum(x[(day, slot)] for slot in range(num_slots))
        # 目標睡眠時間との差分をペナルティとして追加
        penalty_target_sleep += target_sleep_wight * (sleep_sum - target_sleep_time) ** 2
    end_time_target_sleep = time.time()
    print(f"Target Sleep Penalty Calculation Time: {end_time_target_sleep - start_time_target_sleep:.4f} seconds")


    # 寝れない時間帯の場合、強いペナルティを加える
    start_time_cannot_sleep = time.time()

    penalty_cannot_sleep = 0
    for day in range(num_days):
        for slot in range(num_slots):
            if cannot_sleep[day, slot]:
                penalty_cannot_sleep += cannot_sleep_weight * x[(day, slot)]  # 寝れない時間に睡眠を割り当てないように強いペナルティ
    end_time_cannot_sleep = time.time()
    print(f"Cannot Sleep Penalty Calculation Time: {end_time_cannot_sleep - start_time_cannot_sleep:.4f} seconds")

    # 連続睡眠を促進（隣接するスロットを関連付ける）
    start_time_continuous_sleep = time.time()

    penalty_continuous_sleep = 0
    for day in range(num_days):
        for slot in range(num_slots - 1):
            penalty_continuous_sleep -= continuous_sleep_weight * x[(day, slot)] * x[(day, slot + 1)]  # 隣接時間が連続しているほど良い
    end_time_continuous_sleep = time.time()
    print(f"Continuous Sleep Penalty Calculation Time: {end_time_continuous_sleep - start_time_continuous_sleep:.4f} seconds")


    # 90分の倍数のペナルティ（6スロット単位）
    start_time_90min_blocks = time.time()

    penalty_90min_blocks = 0
    for day in range(num_days):
        sleep_count = 0
        for slot in range(num_slots):
            if x[(day, slot)]:
                sleep_count += 1
            else:
                if sleep_count > 0 and sleep_count % 6 != 0:  # 90分単位でない場合にペナルティ
                    penalty_90min_blocks += penalty_90min * (sleep_count % 6)
                sleep_count = 0
    end_time_90min_blocks = time.time()
    print(f"90min Blocks Penalty Calculation Time: {end_time_90min_blocks - start_time_90min_blocks:.4f} seconds")


    # 曜日間の変動を抑制（昨日の同じ時間との関連）
    start_time_day_change = time.time()
    penalty_day_change = 0
    for day in range(num_days - 1):
        for slot in range(num_slots):
            penalty_day_change += day_change_wight * x[(day, slot)] * x[(day + 1, slot)]  # 前日と同じ時間に寝るほど良い
    end_time_day_change = time.time()
    print(f"Day Change Penalty Calculation Time: {end_time_day_change - start_time_day_change:.4f} seconds")


    # 分割数のペナルティ
    start_time_segments = time.time()
    penalty_segments = 0
    for day in range(num_days):
        segment_count = x[(day, 0)] + sum((1 - x[(day, slot - 1)]) * x[(day, slot)] for slot in range(1, num_slots))
        penalty_segments += segments_weight * (segment_count - ideal_segments) ** 2  # 希望分割数との差の二乗ペナルティ
    end_time_segments = time.time()
    print(f"Segments Penalty Calculation Time: {end_time_segments - start_time_segments:.4f} seconds")

    # 連続して起きている時間が4時間以上20時間以内になるように制約を追加
    start_time_awake_gap = time.time()

    penalty_awake_gap = 0
    for day in range(num_days):
        # 1日を跨ぐ可能性も考慮して、前日と翌日のスロットも含めて計算
        awake_count = 0  # 連続して起きている時間のカウント

        # 1日内で起きている時間を計算（スロット1からnum_slots-1まで）
        for slot in range(num_slots):
            if not x[(day, slot)]:  # 起きている時間
                awake_count += 1
            else:  # 寝ている時間が来たらリセット
                if awake_count < 4:  # 起きている時間が4時間未満の場合
                    penalty_awake_gap += (4 - awake_count) * 25  # 4時間未満ならペナルティ
                elif awake_count > 20:  # 起きている時間が20時間を超えている場合
                    penalty_awake_gap += (awake_count - 20) * 15  # 20時間を超えていたらペナルティ
                awake_count = 0  # カウントをリセット

        # 最後のスロットが起きている時間で終わる場合、翌日のスロットと合わせて計算
        if awake_count < 4:
            penalty_awake_gap += (4 - awake_count) * 25  # 起きている時間が4時間未満の場合
        elif awake_count > 20:
            penalty_awake_gap += (awake_count - 20) * 15  # 20時間を超えている場合

        # 前日の起きている時間と今日の起きている時間を合算（跨ぎを考慮）
        if day > 0:  # 1日目以外
            prev_day_awake_count = 0
            for slot in range(num_slots - 1, -1, -1):  # 前日の起きている時間を逆順で計算
                if not x[(day - 1, slot)]:
                    prev_day_awake_count += 1
                else:
                    break

            # 今日の起きている時間と合算
            total_awake_count = prev_day_awake_count + awake_count
            if total_awake_count < 4:
                penalty_awake_gap += (4 - total_awake_count) * 25  # 合算した起きている時間が4時間未満
            elif total_awake_count > 20:
                penalty_awake_gap += (total_awake_count - 20) * 15  # 合算した時間が20時間を超えている場合

    end_time_awake_gap = time.time()
    print(f"Awake Gap Penalty Calculation Time: {end_time_awake_gap - start_time_awake_gap:.4f} seconds")


    # 総コスト関数の設定
    start_time_cost_function = time.time()

    cost_function = (
        penalty_target_sleep +  # 目標睡眠時間に近づける
        penalty_cannot_sleep +   # 寝れない時間帯に寝ない
        penalty_continuous_sleep +  # 連続睡眠を促進
        penalty_90min_blocks +  # 90分ごとの倍数に
        penalty_day_change + # 曜日間の変動を抑制
        penalty_awake_gap +  # 連続して起きている時間が4時間以上20時間以内になるように制約を追加
        penalty_segments  # 分割数のペナルティ
    )
    end_time_cost_function = time.time()
    print(f"Cost Function Calculation Time: {end_time_cost_function - start_time_cost_function:.4f} seconds")

    # QUBOを作成
    start_time_qubo_compile = time.time()
    qubo = cost_function.compile()
    end_time_qubo_compile = time.time()
    print(f"QUBO Compile Time: {end_time_qubo_compile - start_time_qubo_compile:.4f} seconds")

    # サンプラーを使用して最適解を求める
    start_time_sampler = time.time()
    qubo_model = qubo.to_qubo()
    # openjijでSQAを実行
    sampler = oj.SASampler()  # SASampler（Simulated Annealing）を使用
    response = sampler.sample_qubo(qubo_model[0])
    end_time_sampler = time.time()
    print(f"SQA Sampling Time: {end_time_sampler - start_time_sampler:.4f} seconds")

    # 解釈: 睡眠スケジュールを出力
    solution = response.first.sample
    # print("Solution:", solution)  # 解の中身を表示して確認

    # スケジュール表示
    schedule = np.zeros((num_days, num_slots))
    start_time_schedule = time.time()

    for day in range(num_days):
        for slot in range(num_slots):
            key = f"x_{day}_{slot}"
            sleep_value = solution.get(key, 0)
            schedule[day, slot] = sleep_value
            # print(f"Day {day + 1}, Slot {slot}: {sleep_value}")
    end_time_schedule = time.time()
    print(f"Schedule Setup Time: {end_time_schedule - start_time_schedule:.4f} seconds")

    # 最後にスケジュール全体を表示
    # print("Final Sleep Schedule:")
    # print(schedule)

    # 実行時間の計測終了
    total_end_time = time.time()


    # 結果を表示（時間ごとにスケジュールを可視化）
    for day in range(num_days):
        print(f"Day {day + 1}: ", end="")
        
        total_sleep_time = 0  # 合計睡眠時間を初期化
        
        for slot in range(num_slots):
            if schedule[day, slot]:
                # 寝ている時間帯は「█」
                print("█", end="")
                total_sleep_time += 1  # 寝ているスロットをカウント
            else:
                # 寝ていない時間帯は「 」
                print(" ", end="")
        
        # 合計睡眠時間（時間単位で表示）
        total_sleep_hours = total_sleep_time / 4  # 1スロットは15分なので、時間に換算
        print(f" | Total sleep time: {total_sleep_hours:.2f} hours")  # 合計睡眠時間を表示

    # 実行時間を表示
    if not (total_start_time is None or total_end_time is None):
        print(f"Execution Time: {total_end_time - total_start_time:.4f} seconds")
    else:
        print("Error: start_time or end_time is None")


    return schedule