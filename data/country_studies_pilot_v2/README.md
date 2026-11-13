# Country Studies 실자료 파일럿 v2

한국·북한·일본·중국 원문 4권의 추출 텍스트와 한국어 질문 40문항을 프로젝트에 함께 포함합니다. 저장소를 받은 뒤 별도 PDF·ZIP 복사 없이 검증을 실행할 수 있습니다. v1의 기존 30문항은 유지하고 중국편 10문항을 추가했습니다. 실제 모델의 성능 측정 결과는 아닙니다.

## 실행

프로젝트 루트에서 Python 의존성을 설치한 뒤 검증합니다.

```bash
python -m pip install -e ".[dev]"
python -m rag_eval validate --config configs/country_studies_pilot_v2_dev.yaml
python -m rag_eval validate --config configs/country_studies_pilot_v2_test.yaml
```

Ollama가 실행 중이고 `bge-m3`가 준비되어 있으면 개발용 검색 실험을 시작합니다.

```bash
python -m rag_eval run --config configs/country_studies_pilot_v2_dev.yaml -v
```

기본은 검색 전용입니다. 개발용으로 설정을 정한 뒤 최종 평가용을 실행하고, 답변 생성·거부 정확도까지 보려면 답변 모델을 별도로 지정합니다.

## 데이터 범위

| 원문 | 판권면 연도 | PDF 쪽수 | 질문 split | 문항 |
| --- | ---: | ---: | --- | ---: |
| South Korea: A Country Study, 4판 | 1992 | 462 | dev | 10 |
| North Korea: A Country Study, 5판 | 2008 | 384 | dev | 10 |
| Japan: A Country Study, 5판 | 1992 | 670 | test | 10 |
| China: A Country Study, 4판 | 1988 | 788 | test | 10 |
| 합계 | | 2,304 | dev 20 / test 20 | 40 |

사실·수치·비교·복수 근거·답변 불가가 각각 8문항입니다. 답변 가능한 32문항에 영문 근거 42개를 연결했습니다. 검색 회수율의 분모는 전체 40개가 아닌 32개이며, 개발용과 최종 평가용에 각각 16개가 있습니다.

원문 4권 모두 검색 대상에 둡니다. 질문만 출처 그룹에 따라 나눴습니다. 질문은 국가안보 본문의 선택된 부분을 중심으로 작성했으므로 책 전체 주제나 일반 한국어 RAG 성능을 대표하지 않습니다. 영어 문서를 한국어 질문으로 검색하는 파일럿입니다.

## 파일과 검증

- `docs/*.txt`: 업로드 PDF에서 pypdf로 추출한 본문. NFC·줄바꿈만 통일하며 단어 분절·OCR 철자·머리말·쪽수는 자동 교정하지 않았습니다. 이미지 자체는 포함하지 않습니다.
- `qa.jsonl`: 질문·정답·유형·split과 원문 근거. `pdf_page`는 1부터 시작하는 PDF 파일 쪽수이며 인쇄 쪽수와 다릅니다.
- `sources.jsonl`: 원본 파일명, 공식 서지 링크, 원본 PDF·추출 텍스트 SHA-256, 추출 버전과 권리 안내.
- `page_index.json`: PDF 쪽수와 추출 텍스트 위치 대응.
- `validation.json`: 근거 일치·쪽수 대응·청크 도달 가능성의 검증 기록. 모델 검색 점수는 아닙니다.

근거가 연결되는 PDF 19개 쪽을 원문 이미지와 대조하고, 두 청크 설정(1000/200·800/100)에서 각 근거를 50% 이상 덮는 청크가 존재하는지 확인했습니다. 실제로 상위 k개에 검색되었다는 뜻은 아닙니다.

## v1과 비교할 때

v1은 3권·30문항, v2는 4권·40문항입니다. v1의 질문·정답·근거를 바꾸지 않았으며, v1의 추출 본문도 별도 `data/country_studies_pilot_v1/docs/`에 동일한 내용으로 포함합니다. 다만 v2는 검색 대상과 질문 수가 늘었으므로 같은 데이터 조건의 점수로 섞지 않습니다.

답변 불가 8문항은 모두 판본 발행 이후인 2026년의 실제 관측치를 요구하는 쉬운 부정 예시입니다. 어려운 부정 예시와 더 넓은 주제의 질문은 후속 보강 대상입니다. 서술은 각 판본 기준이며 현재 군사·행정·법률 현황을 뜻하지 않습니다.

추출 본문의 출처는 [미 의회도서관 Country Studies](https://www.loc.gov/collections/country-studies/about-this-collection/)입니다. [권리 안내](https://www.loc.gov/collections/country-studies/about-this-collection/rights-and-access/)는 일부 사진의 별도 권리와 나머지 텍스트·그래픽의 퍼블릭 도메인을 설명합니다. 본 데이터는 추출 텍스트만 포함하고 원본 PDF나 사진 파일을 배포하지 않습니다. 중국편 판권면의 정부 저작권 문구와 컬렉션 안내의 차이도 `sources.jsonl`에 기록했습니다.
