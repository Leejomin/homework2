실험 요약<br>
실험 결과, 고채도 작물은 HSV 기법만으로도 충분한 정확도를 확보할 수 있으나, 저채도 작물이나 배경의 질감이 복잡한 환경에서는 HSV 기법을 주 알고리즘으로 채택하고 Adaptive 기법의 에지 정보를 보조적으로 결합하는 융합형 알고리즘이 필요할 것으로 보인다.

1.실험 목적<br>
본 실험은 그림자 및 반사광 등 외부 조명 변수가 존재하는 환경에서 Adaptive Thresholding과 HSV Segmentation 기법의 조명 변화 적응성을 정량적으로 비교 분석하는 것을 목적으로 한다. 이를 통해 실제 수확 현장에서의 비전 알고리즘 적용 가능성을 검토하고자 한다.

2.핵심 알고리즘 구현 원리<br>
2.1. Adaptive Thresholding (적응형 이진화)<br>
이미지의 국소 영역마다 임계값을 동적으로 계산하여 이진화하는 기법이다. 각 픽셀 $(x, y)$에 대한 임계값 $T(x, y)$는 주변 블록 영역의 가우시안 가중치 평균에서 상수 $C$를 뺀 값으로 결정된다.

$$T(x, y) = \text{mean}(I_{block}) - C$$


```Python
ada = cv2.adaptiveThreshold(gray, 255, 
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY_INV, 51, 15)
```

동작 특징: 전체 조도가 변하더라도 주변 픽셀과의 명암 차이를 이용하므로 그림자 영역 내에서도 물체를 식별할 수 있다. 다만, 배경의 텍스처(나뭇결 등)를 형태 정보로 오인할 가능성이 있다.

2.2. HSV Segmentation (색 공간 분할)<br>
명도(Value)의 간섭을 최소화하기 위해 색상(Hue)과 채도(Saturation) 영역에서 물체를 추출하는 기법이다.

```Python
mask1 = cv2.inRange(hsv, config["apple_red_low1"], config["apple_red_upp1"])
mask2 = cv2.inRange(hsv, config["apple_red_low2"], config["apple_red_upp2"])
hsv_mask = cv2.bitwise_or(mask1, mask2)
```
동작 특징: 조명 강도가 변해도 물체 고유의 색상 정보를 추적하므로 그림자 내성에 최적화되어 있다. 단, 표면 반사광에 의해 색상 정보가 유실될 경우 마스크 내부 결손이 발생할 수 있으므로 임계값의 정밀한 조정이 필요하다.

3.실험 결과 및 데이터 분석<br>
3.1. 성능 평가 지표 및 판단 기준<br>

알고리즘의 신뢰성을 정량화하기 위해 다음 수식을 사용한다.

편향 (Bias): 기준 면적($A_{base}$) 대비 실험 면적($A_{test}$)의 오차율

$$\text{Bias} (\\%) = \frac{A_{test} - A_{base}}{A_{base}} \times 100$$

Positive (+): 배경 노이즈 등으로 인한 과다 산출 (Over-segmentation)

Negative (-): 그림자나 결손으로 인한 과소 산출 (Under-segmentation)

정확도 (Accuracy): 기준 면적과의 일치율

$$\text{Accuracy} (\\%) = \left( 1 - \left| \frac{A_{test} - A_{base}}{A_{base}} \right| \right) \times 100$$

3.3. 사례별 심층 분석사례<br>

3.3.1. 사과 (Apple)<br>

![](/image/apple_analysis_result.png)
고채도 작물사과는 배경과의 색상 대비가 뚜렷하여 HSV 색 공간에서 분리가 용이한 고대비(High Contrast) 특성을 가진다.

HSV (Acc: 95.9%): 조명 변화를 효과적으로 극복하여 높은 신뢰도를 유지했다. 다만, 사과 표면의 정반사(Specular Reflection) 현상으로 인해 마스크 내부에 국소적인 결손이 발생하였다.

Adaptive (Acc: 90.3%): 그림자 영역에서 임계값 변화로 인해 하단 실루엣이 소실되는 음수 편향(-9.7%)이 관찰되었다.


3.3.2. 아보카도 (Avocado)<br>

![](/image/avocado_analysis_result.png)

저채도 작물아보카도는 어두운 톤의 나뭇결 배경과 명도/채도 값이 유사한 저대비(Low Contrast) 특성을 가진다.

HSV (Acc: 90.9%): 배경 노이즈 제거 능력은 우수하나, 아보카도 고유의 어두운 색상이 임계 범위 경계에 위치하여 사과 대비 내부 결손율이 높게 나타났다.

Adaptive (Acc: 85.1%): 나뭇결의 미세한 질감을 '에지'로 인식하여 물체 외곽에 다량의 노이즈가 포함되는 배경 질감 간섭 현상이 두드러졌다.

3.4. 배경 및 조명 특성 분석<br>
3.4.1. 나뭇결 질감의 영향<br>
Adaptive 기법은 국소 영역의 명암 차에 민감하여, 나뭇결처럼 좁은 간격의 명암 변화를 물체의 실루엣으로 오인하여 Positive Bias를 유발한다.

3.4.2. 그림자의 영향<br>
그림자는 명도($V$)를 낮추지만 색상($H$)은 유지시킨다. 이는 HSV 기법이 그림자 환경에서 높은 신뢰성을 보이는 근거가 된다.

4.결론<br>
실험 결과, 고채도 작물은 HSV 기법만으로 충분한 정확도를 확보할 수 있으나 저채도 작물이나 배경 질감이 복잡한 환경에서는 HSV 기법을 주 알고리즘으로 채택하고 Adaptive 기법의 경계(Edge) 정보를 결합하는 방식이 인식도 향상에 유리하다. 특히 Adaptive 기법은 작물의 정밀한 외곽 경계를 검출하는 데 강점이 있어 필터링 기술과 병행 시 정교한 좌표 계산에 활용될 가치가 높다.
