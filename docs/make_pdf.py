"""
INN 픽셀 수준 동작 원리 PDF 생성 스크립트
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 한글 폰트 등록 ─────────────────────────────────────────────────
FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕 (일반)
    "C:/Windows/Fonts/malgunbd.ttf",    # 맑은 고딕 Bold
    "C:/Windows/Fonts/NanumGothic.ttf",
]
FONT_BOLD_PATHS = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
]

def reg(name, paths):
    for p in paths:
        if Path(p).exists():
            pdfmetrics.registerFont(TTFont(name, p))
            return True
    return False

reg("Korean",     FONT_PATHS)
reg("KoreanBold", FONT_BOLD_PATHS)

# ── 스타일 ─────────────────────────────────────────────────────────
W, H = A4
LEFT = RIGHT = 18*mm

def make_styles():
    base = getSampleStyleSheet()
    def s(name, **kw):
        kw.setdefault("fontSize", 10)
        kw.setdefault("leading", 16)
        return ParagraphStyle(name, fontName="Korean", **kw)
    return {
        "title":   ParagraphStyle("title",   fontName="KoreanBold", fontSize=18,
                                  leading=26, spaceAfter=6, textColor=colors.HexColor("#1a237e")),
        "h1":      ParagraphStyle("h1",      fontName="KoreanBold", fontSize=13,
                                  leading=20, spaceBefore=14, spaceAfter=4,
                                  textColor=colors.HexColor("#283593")),
        "h2":      ParagraphStyle("h2",      fontName="KoreanBold", fontSize=11,
                                  leading=18, spaceBefore=10, spaceAfter=3,
                                  textColor=colors.HexColor("#3949ab")),
        "body":    s("body",   spaceAfter=6),
        "code":    ParagraphStyle("code",    fontName="Korean",     fontSize=8.5,
                                  leading=14, backColor=colors.HexColor("#f5f5f5"),
                                  leftIndent=8, rightIndent=8, spaceAfter=6,
                                  borderPadding=(4,4,4,4)),
        "note":    s("note",   fontSize=9, textColor=colors.HexColor("#555555"),
                     leftIndent=12, spaceAfter=4),
        "caption": s("caption", fontSize=9, textColor=colors.HexColor("#888888"),
                     alignment=1),
    }

def code_block(text, st):
    lines = text.strip().split("\n")
    safe = "<br/>".join(
        l.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        for l in lines
    )
    return Paragraph(safe, st["code"])

def kpi_table(data, st):
    """헤더 + 데이터 행으로 구성된 표"""
    header_bg = colors.HexColor("#283593")
    alt_bg    = colors.HexColor("#e8eaf6")
    style = TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), header_bg),
        ("TEXTCOLOR",   (0,0),(-1,0), colors.white),
        ("FONTNAME",    (0,0),(-1,0), "KoreanBold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("FONTNAME",    (0,1),(-1,-1), "Korean"),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, alt_bg]),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("GRID",        (0,0),(-1,-1), 0.4, colors.HexColor("#9fa8da")),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ])
    col_w = (W - LEFT - RIGHT) / len(data[0])
    t = Table(data, colWidths=[col_w]*len(data[0]))
    t.setStyle(style)
    return t

# ── 본문 구성 ──────────────────────────────────────────────────────
def build_story(st):
    story = []
    SP = lambda n=6: Spacer(1, n)
    HR = lambda: HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#9fa8da"), spaceAfter=4)

    # ── 제목 ─────────────────────────────────────────────────────
    story += [
        Paragraph("INN 픽셀 수준 동작 원리", st["title"]),
        Paragraph("SecureFace-RX / PRO-Face S 기술 설명", st["body"]),
        HR(), SP(8),
    ]

    # ── 0. 핵심 요약 ─────────────────────────────────────────────
    story += [
        Paragraph("핵심 요약", st["h1"]),
        Paragraph(
            "보호본(ŷ)은 블러(y)와 픽셀 평균 <b>0.21px</b> 차이로 육안 구분이 불가능합니다. "
            "그러나 이 미세한 차이 안에 원본을 복원하는 수식의 분자가 숨겨져 있으며, "
            "키(K) 없이는 수식 자체가 성립하지 않아 복원이 불가능합니다.",
            st["body"]
        ),
        SP(),
    ]

    # ── 1. 전제: DWT ────────────────────────────────────────────
    story += [
        Paragraph("1. 전제 — DWT(하르 웨이블릿 변환)", st["h1"]),
        Paragraph(
            "INN은 픽셀값이 아닌 <b>웨이블릿 계수</b>로 작동합니다. "
            "2×2 픽셀 패치를 4개의 주파수 성분으로 분해합니다.",
            st["body"]
        ),
        SP(4),
        kpi_table([
            ["계수", "의미", "원본 예시값", "블러 후"],
            ["LL", "전체 평균 밝기 (저저)", "355", "354  (거의 유지)"],
            ["LH", "수평 방향 엣지 (저고)", "25",  "0  (사라짐)"],
            ["HL", "수직 방향 엣지 (고저)", "5",   "0  (사라짐)"],
            ["HH", "대각선 엣지·노이즈 (고고)", "15", "0  (사라짐)"],
        ], st),
        SP(4),
        Paragraph(
            "블러의 핵심: <b>고주파 계수(LH, HL, HH)를 0으로 지운다.</b>  "
            "원본 복원 = 이 고주파 계수들을 되살린다.",
            st["note"]
        ),
        SP(),
    ]

    story += [
        Paragraph("DWT 수식 (하르 웨이블릿, 2×2 패치)", st["h2"]),
        code_block(
            "원본 x:  [[200, 180],    →   LL = (200+180+160+170)/2 = 355\n"
            "         [160, 170]]         LH = (200+180-160-170)/2 =  25\n"
            "                             HL = (200-180+160-170)/2 =   5\n"
            "                             HH = (200-180-160+170)/2 =  15\n"
            "\n"
            "블러 y:  [[177, 177],    →   LL = 354,  LH = 0,  HL = 0,  HH = 0\n"
            "         [177, 177]]         (블러 = 엣지 성분 제거)",
            st
        ),
        SP(),
    ]

    # ── 2. INN Forward ────────────────────────────────────────
    story += [
        Paragraph("2. INN 보호 단계 (Affine Coupling)", st["h1"]),
        Paragraph("INN은 입력을 두 덩어리로 나눕니다:", st["body"]),
        code_block(
            "x1 = DWT(원본) 전체  = [LL=355, LH=25, HL=5, HH=15]  ← 원본의 4개 계수\n"
            "x2 = DWT(블러) 전체  = [LL=354, LH= 0, HL=0, HH= 0]  ← 블러의 4개 계수",
            st
        ),
        Paragraph("Affine Coupling 연산:", st["h2"]),
        code_block(
            "s, t = 서브넷(x2, K)           ← 블러 계수 + 키로 변환값 계산\n"
            "\n"
            "네트워크가 학습으로 찾아낸 s, t (각 계수별):\n"
            "  LL: exp(s_LL) × 355 + t_LL ≈ 354  →  t_LL ≈  -1\n"
            "  LH: exp(s_LH) ×  25 + t_LH ≈   0  →  t_LH ≈ -25\n"
            "  HL: exp(s_HL) ×   5 + t_HL ≈   0  →  t_HL ≈  -5\n"
            "  HH: exp(s_HH) ×  15 + t_HH ≈   0  →  t_HH ≈ -15\n"
            "\n"
            "y1 = exp(s) × x1 + t ≈ [354, 0, 0, 0] = DWT(블러)\n"
            "\n"
            "→ IWT 후  ŷ ≈ 177.002  (블러 y=177과 0.002 차이, 저장됨)",
            st
        ),
        SP(),
    ]

    # ── 3. 복원 ───────────────────────────────────────────────
    story += [
        Paragraph("3. 복원 단계 (Affine Coupling 역산)", st["h1"]),
        code_block(
            "저장된 y1 ≈ [354, 0, 0, 0]과 키 K로 역산:\n"
            "\n"
            "x1_복원 = (y1 - t) / exp(s)\n"
            "        = ([354, 0, 0, 0] - [-1, -25, -5, -15]) / exp(≈0)\n"
            "        = [355, 25, 5, 15]  =  DWT(원본)  ← 완전 복원!\n"
            "\n"
            "→ IWT 후  [[200, 180], [160, 170]]  원본 픽셀 복원",
            st
        ),
        SP(),
    ]

    # ── 4. 틀린 키 ───────────────────────────────────────────
    story += [
        Paragraph("4. 틀린 키로 복원 시도", st["h1"]),
        code_block(
            "K_wrong 사용 시:\n"
            "  s_wrong = 0.8,  t_wrong = 100   (완전히 다른 값)\n"
            "\n"
            "x1_wrong = (y1 - t_wrong) / exp(s_wrong)\n"
            "         = (0.45 - 100) / exp(0.8)\n"
            "         = -99.55 / 2.23\n"
            "         = -44.6   ← 쓰레기 (원본 = 25)",
            st
        ),
        SP(),
    ]

    # ── 5. 실제 측정 결과 ────────────────────────────────────
    story += [
        Paragraph("5. 실제 얼굴 이미지 측정 결과 (chacha.jpg)", st["h1"]),
        Paragraph("실제 모델로 256×256 얼굴 크롭을 처리한 수치입니다.", st["body"]),
        SP(4),
    ]

    regions = [
        ("이마 (밝은 영역)",
         "66  72 / 82  85", "113 108 / 111 106", "113 108 / 111 106", "65  72 / 82  84",
         "LL=+66.5, LH=+16.5, HL=+9.5, HH=+1.5", "0.50", "0.000"),
        ("눈 주변 (중간 영역)",
         "121 107 / 112  96", "145 146 / 146 146", "145 146 / 146 146", "121 106 / 111  96",
         "LL=+73.5, LH=-10.5, HL=-15.5, HH=+0.5", "0.50", "0.000"),
        ("눈동자 (어두운 영역)",
         "210 188 / 203 180", "156 156 / 155 156", "156 156 / 155 156", "209 187 / 203 179",
         "LL=-79.0, LH=-7.0, HL=-23.0, HH=+1.0", "0.75", "0.000"),
        ("윤곽선 (엣지 영역)",
         "237 237 / 237 237", "224 222 / 223 221", "224 221 / 223 221", "236 236 / 236 236",
         "LL=-29.5, LH=+0.5, HL=+2.5, HH=+0.5", "1.00", "0.250"),
        ("배경과 경계",
         "39  39 / 36  38", "40  42 / 39  41", "40  42 / 39  41", "39  39 / 36  39",
         "LL=+5.0, LH=-1.0, HL=-1.0, HH=-1.0", "0.25", "0.000"),
    ]

    for name, orig, blur, prot, rest, t_vals, rec_err, prot_err in regions:
        story += [
            Paragraph(f"[ {name} ]", st["h2"]),
            kpi_table([
                ["", "원본 픽셀", "블러 픽셀", "보호본 픽셀", "복원 픽셀"],
                ["2×2 값", orig, blur, prot, rest],
            ], st),
            SP(3),
            Paragraph(f"학습된 이동량 t : {t_vals}", st["note"]),
            Paragraph(
                f"복원 오차: <b>{rec_err}px</b> / 보호 위장 오차: <b>{prot_err}px</b>",
                st["note"]
            ),
            SP(6),
        ]

    # ── 6. 전체 통계 ──────────────────────────────────────
    story += [
        Paragraph("6. 전체 이미지 통계 (256×256)", st["h1"]),
        SP(4),
        kpi_table([
            ["항목", "평균 차이", "최대 차이", "의미"],
            ["블러 - 원본",   "24.06 px", "175 px", "블러가 이미지를 이 정도 바꿈"],
            ["보호본 - 블러",  " 0.21 px", "  4 px", "숨겨진 신호 강도 (육안 불가)"],
            ["복원 - 원본",   " 0.60 px", " 32 px", "복원 오차 (사실상 완벽)"],
        ], st),
        SP(6),
        Paragraph(
            "사람 눈이 색상 차이를 감지하는 한계: 약 3~5px. "
            "보호본의 0.21px 차이는 그 15배 이하로 절대 구분 불가능합니다.",
            st["note"]
        ),
        SP(),
    ]

    # ── 7. 핵심 정리 ──────────────────────────────────────
    story += [
        Paragraph("7. 핵심 정리", st["h1"]),
        SP(4),
        kpi_table([
            ["항목", "설명"],
            ["저장되는 것",  "ŷ (보호본) — 블러처럼 보이지만 픽셀값이 0.002 다름"],
            ["원본 저장 여부", "없음 — 원본은 어디에도 저장되지 않음"],
            ["복원 정보 위치", "ŷ 픽셀값 자체 (수식 x = (ŷ - t) / exp(s)의 분자)"],
            ["키 K의 역할",  "분모 exp(s)와 이동량 t를 결정"],
            ["키 없이 복원 시", "s, t가 완전히 달라져 쓰레기 값 산출 (PSNR≈5dB)"],
        ], st),
        SP(8),
        Paragraph(
            "한 줄 요약: ŷ의 픽셀값이 수식 x = (ŷ - t) / exp(s) 의 분자이고, "
            "키 K가 분모(exp(s))와 이동량(t)을 결정한다. 키 없이는 수식 자체가 성립하지 않는다.",
            st["body"]
        ),
        SP(4),
        Paragraph(
            "참고: 실제 모델은 3채널 256×256 이미지 전체에 3개의 Affine Coupling Block을 반복 적용하며, "
            "위 예시는 1채널 1블록의 단순화 버전입니다. 관련 코드: models/invblock.py",
            st["note"]
        ),
    ]

    return story


def main():
    out = Path(__file__).parent / "INN_pixel_level_explanation.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=18*mm, bottomMargin=18*mm,
    )
    st = make_styles()
    doc.build(build_story(st))
    print(f"저장 완료: {out.resolve()}")


if __name__ == "__main__":
    main()
