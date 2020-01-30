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



*preprocessing.py 에서 input feature 를 뽑아내는 코드를 다시 작성중
    - iteration 을 face 단위로 도는 것이 좋을지 edge 단위로 도는 것이 좋을지... 
        - vtkExtractEdges 를 사용해서 edge 단위로 iteration 돌려고 했더니.. point ordering 이 달라져있어서 제대로 안됨..
    - 속도도 중요하지만 코드를 최대한 직관적으로 짜서  c++ 에 이식하기 쉽도록 만들지