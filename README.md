Mesh() 에 vtkpolydata 를 parameter 로 받게는 했음
- manifolds 없는 깨끗한 데이터, face 같은것도 다 있어서,,ㄴ ㅏ중에 다른 데이터 확인 필요
- visualize 어떻게하는지 봐야할듯 (finished)
- mesh prepare.py 에서 build_gemm 과 extracxt_features() 가 정확히 뭘하는지 봐야함

- gemm_edge 포함 모든 Mesh Class 내부 구조 해체
    - Mesh 에 들어있는 내용들을 전부 Network 안에 넣어줘야함....
    - 할수있을지..


- pre-trained network 돌려서 visualize 해보기 (완료)
    - pre-trained model 을 torch script model 로 저장하기 : Mesh class 를 input 으로 받기때문에 현재로써는 불가능
    - torch script model load 해서 visualize 해보기

