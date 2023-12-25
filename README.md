<div align="center">
<h2>[2022] 인공지능 온라인 경진대회 - 상수관로 누수감지 및 분류 문제</h2>
색다른 Normalization 방법을 통해 예측 정확도를 극대화한 이상 감지 및 분류 프로젝트 🚀
</div><br/> 

## 개요

- 프로젝트 이름: 2022 인공지능 온라인 경진대회 - 상수관로 누수감지 및 분류 문제 :fountain:
- 프로젝트 지속기간: 2022.06-2022.08
- 개발 언어 및 툴: Python & TensorFlow
- 작업 인원: 1명<br/><br/>

## 프로젝트 설명

<center>
</center>

과학기술정보통신부가 주최한 2022 인공지능 온라인 경진대회에서 4위 이내로 입상하여 사업화 지원 기업에 선정되는 것을 목표로 프로젝트를 진행하였습니다. 최종 사업화 지원 기업에는 선정되지 못했지만, 수치해석 분야에서 3위를 달성하였습니다.


- 프로젝트 소개 :eyes:
	- 상수관로 진동 센서 데이터로 누수 유형을 분류하는 문제<br/><br/>
- 기대 효과 :dart:
	- 전국 수도관의 13%가 30년 이상 된 노후관로이며, 이는 상수도 품질 저하의 주요 원인
	- 상수관로 누수 감지 및 분류를 자동화하여 시간과 비용을 절감할 수 있음<br/><br/>
- 데이터 소개 :bar_chart:
	- 독립 변수(Independent Variable)
		- 센서 출력값에 Fourier Transform을 적용하여 계산한 주파수 별 Spectral Density 값
		- 0Hz부터 5120Hz까지 10Hz단위로 수집된 513개 Columns
	- 종속 변수(Dependent Variable)
		- 누수 구분 클래스(Leaktype)
		- 옥외누수(Out), 옥내누수(In), 정상(Normal), 전기/기계음(Noise), 환경음(Other)
		**이미지첨부하기**<br/><br/>
- 진행과정 요약 :chart_with_upwards_trend:
	- 데이터 EDA 및 시각화
	- 클래스 불균형 문제로 인한 SMOTE 오버샘플링 기법 적용
	- <strong>두 가지 정규화 방법(Column Based, Row Based)</strong>을 통해 데이터로부터 얻을 수 있는 정보를 극대화
	- KNeighborsClassifier, Conv1D 등 전통적인 머신러닝 모델과 딥러닝 모델을 모두 사용하여 최종 모델 구성
	- 최종 성능 F1 Score 0.92로 수치해석 분야 3위에 입상
	- **이미지첨부하기**<br/><br/>

</div>
