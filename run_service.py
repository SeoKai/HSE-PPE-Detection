import os, sys, types, runpy
# 작업 스케줄러(SYSTEM) 세션에는 오디오 장치가 없어 mediapipe가 불러오는 sounddevice 초기화가 실패한다. 오디오는 쓰지 않으므로 빈 모듈로 대체.
sys.modules['sounddevice'] = types.ModuleType('sounddevice')
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py'), run_name='__main__')
