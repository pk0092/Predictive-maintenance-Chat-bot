Predictive Maintenance(예측 보전)을 목표로,
설비 고장 발생시 조치 사항을 추천해주는 프로그램입니다.

1. 고장 현상에 대한 조치 사항 Labelling
2. RNN Deep Learning 이용(HGTK, LSTM, Keras_Optimizer)
  고장 현상 내용을 자소단위로 분할, 한글토큰(hgtk)을 이용해 인풋 재 정의 및 One-hot vector로 변환
  조치 사항 Label을 Token화
  인풋에 대한 Output 연관관계 Optimize
3. 신규 인풋에 대한 가장 관계있는 Top-5 token 출력
4. Web service Deploy
 