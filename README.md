# ML_forEveryone
## 소개
구글에서 진행하는 '머신러닝 스터디 잼'의 그룹 '황박사와 아이들' 팀의 
기초반 이후 extra 스터디 (파이선을 이용한 머신러닝 및 MNIST) 소스코드입니다.

## 파일 소개
1. 기본 문법
2. 선형 회귀
3. 최소의 손실값 구하기
* 평균 제곱 오차(MSE)를 이용한 함수
* 도함수를 이용한 손실 함수
4. 다중 입력값에서의 선형 회귀
* 직접 변수 선언해서 실행
* matrix와 matmul을 이용한 연산
* 입력값을 파일로부터 받아서 연산
* TextLineReader를 이용한 배치 실행
5. 로지스틱 회귀
* 시그모이드를 이용한 로지스틱 회귀 (로그 손실 : tf.reduce_mean( -y * log(y`) - (1-y)log(1-y1) ))
* 입력값을 파일로부터 받아서 연산
6. 소프트맥스를 이용한 분류
* softmax = exp(logits) / reduce_sum(exp(logits), dim)
입력값을 파일로부터 받아서 연산
7. MNIST
* softmax와 Neural Network를 이용한 학습
  * 소프트맥스 이용 -> 89.51% 정확도
  * with Adam Optimizer -> 90.23% 정확도
  * Nerual Network와 xavier initializer 이용 -> 97.99% 정확도
* Deep Neural Network를 이용한 학습
  * 5 depth를 이용한 학습 -> 97.72% 정확도 (overfitting으로 인한 정확도 낮아짐)
  * dropout을 이용한 Deep Neural Network -> 98.39% 정확도
* CNN을 이용한 학습 -> 99.33% 정확도
