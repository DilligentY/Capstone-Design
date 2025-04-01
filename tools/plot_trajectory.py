from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

logdir = os.path.join(os.getcwd(), "logs", "skrl", 'quadcopter_direct')
run_dirs = [d for d in glob.glob(os.path.join(logdir, '*')) if os.path.isdir(d)]

all_data = {}

for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)  # 예) run1, run2 등 폴더명
    ea = event_accumulator.EventAccumulator(run_dir)
    ea.Reload()
    
    # 이 폴더(실험)에 있는 모든 태그 정보 확인
    tags = ea.Tags()
    scalar_tags = tags.get('scalars', [])

    # 각 태그별로 스칼라 값을 가져와서 저장
    run_data = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        values = [e.value for e in scalar_events]
        steps = [e.step for e in scalar_events]
        
        run_data[tag] = {
            'steps': steps,
            'values': values
        }
    
    # 모든 태그를 모아서 all_data에 저장
    all_data[run_name] = run_data

# 이제 all_data 딕셔너리에 각 실험(run)별, 태그별로 분류된 데이터가 담겨 있음
print(all_data.keys())  # dict_keys(['run1', 'run2', 'run3', ...])

rew_1 = None
rew_2 = None
rew_3 = None
plt.figure(figsize=(12, 6))

def get_interp(values, steps, rew, np_values):
    common_steps_pre = np.linspace(0, 4800, np_values.shape[0])
    if rew is None:
        rew = np_values
    else:
        try:
            rew = np.hstack([rew, np_values])
        except:
            common_steps_pre = np.linspace(0, 4800, rew.shape[0])
            interpolated_values = np.interp(common_steps_pre, steps, values).reshape(-1, 1)
            print("size mismatch")
            rew = np.hstack([rew, interpolated_values])

    return rew, common_steps_pre

for run_name, run_data in all_data.items():
    parts = run_name.split('_')
    part_hour = parts[1].split('-')[0]
    part_hour_int = int(part_hour)
    steps = run_data["Reward / Total reward (mean)"]['steps']
    values = run_data["Reward / Total reward (mean)"]['values']
    np_values = np.array(values).reshape(-1, 1)

    if part_hour_int < 12:
        rew_1, common_steps_1 = get_interp(values, steps, rew_1, np_values)
        plt.plot(steps, values, alpha=0.1, color='red')

    elif 12 < part_hour_int < 16:
        rew_2, common_steps_2 = get_interp(values, steps, rew_2, np_values)
        plt.plot(steps, values, alpha=0.1 , color="blue")
    else:
        rew_3, common_steps_3 = get_interp(values, steps, rew_3, np_values)
        plt.plot(steps, values, alpha=0.1, color="green")

mean_1 = np.mean(rew_1, axis=1)
mean_2 = np.mean(rew_2, axis=1)
mean_3 = np.mean(rew_3, axis=1)

plt.plot(common_steps_1, mean_1, label='rollout=24', color='red')
plt.plot(common_steps_2, mean_2, label='rollout=36', color='blue')
plt.plot(common_steps_3, mean_3, label='rollout=48', color='green')
plt.title('Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.grid()
plt.show()

