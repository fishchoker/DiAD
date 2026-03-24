# MVTec-AD 15个类别的特定提示词 (每类3个)
CATEGORY_PROMPTS = {
    'bottle': [
        "a photo of the bottle without damage for anomaly detection.",
        "a close-up photo of a bottle.",
        "a close-up photo of a bottle without damage."
    ],
    'cable': [
        "a photo of a unblemished cable for visual inspection.",
        "a close-up photo of the cable without damage.",
        "a close-up photo of a unblemished cable"
    ],
    # 'capsule': [
    #     "a photo of a small capsule without flaw.",
    #     "a photo of the small capsule without flaw.",
    #     "a close-up photo of a capsule without flaw."
    # ],
    'capsule': [
    # 强调胶囊表面光滑完整（针对 crack / scratch / poke）
    "a close-up photo of a capsule with smooth and intact surface.",
    # 强调印字清晰规则（针对 faulty imprint）
    "a close-up photo of a capsule with clear and regular imprint markings.",
    # 强调整体形状饱满无变形（针对 squeeze）
    "a close-up photo of a capsule with uniform shape and no deformation.",
    ],
    'carpet': [
        "a photo of the carpet without flaw for anomaly detection.",
        "a cropped photo of the carpet without defect.",
        "a photo of the carpet for anomaly detection"
    ],
    # 'grid': [
    #     "a close-up photo of a grid without damage.",
    #     "a close-up photo of a grid without defect.",
    #     "a cropped photo of a grid without defect."
    # ],
    'grid': [
    # 强调网格的规则性和均匀性（bent/broken 异常破坏的核心特征）
    "a close-up photo of a grid with uniform and regular mesh pattern.",
    # 强调线条完整连续（针对 broken / thread 类异常）
    "a close-up photo of a grid with continuous and intact grid lines.",
    # 强调表面无外来污染（针对 glue / metal_contamination 类异常）
    "a close-up photo of a clean grid surface without any contamination.",
    ],
    'hazelnut': [
        "a cropped photo of a hazelnut.",
        "a cropped photo of a hazelnut without damage.",
        "a photo of a hazelnut without damage for anomaly detection."
    ],
    'leather': [
        "a cropped photo of a leather.",
        "a cropped photo of a leather without damage.",
        "a close-up photo of a leather without damage."
    ],
    # 'metal_nut': [
    #     "a photo of a metal nut without defect for visual inspection.",
    #     "a close-up photo of a metal nut without defect.",
    #     "a photo of a unblemished metal nut for visual inspection."
    # ],
    # 'metal_nut': [
    # # 强调螺纹完整清晰（针对 thread / bent）
    # "a close-up photo of a metal nut with complete and well-defined threads.",
    # # 强调表面颜色均匀无氧化（针对 color 变色异常）
    # "a close-up photo of a metal nut with uniform metallic surface color.",
    # # 强调正向朝上、方向正确（针对 flip 翻转异常）
    # "a close-up photo of a metal nut placed correctly with visible thread hole.",
    # ],
    'metal_nut': [
        # 整体状态描述，贴近自然语言
        "a close-up photo of a metal nut in perfect condition.",
        # 针对 scratch / bent，用自然语言描述表面
        "a close-up photo of a metal nut with clean and undamaged surface.",
        # 针对 flip（朝向异常），用自然语言描述
        "a close-up photo of a metal nut oriented correctly without any deformation.",
    ],
    'pill': [
        "a photo of a unblemished pill for visual inspection.",
        "a photo of a pill without flaw for visual inspection.",
        "a photo of the pill without flaw for visual inspection."
    ],
    # 'screw': [
    #     "a photo of a small screw.",
    #     "a photo of a small unblemished screw.",
    #     "a photo of a small screw without damage."
    # ],
    # 'screw': [
    # # 强调螺纹完整性（screw 最核心的正常特征）
    # "a close-up photo of a screw with intact threads and clean surface.",
    # # 强调表面无缺陷（针对 scratch / manipulated 类异常）
    # "a close-up photo of a screw with smooth and undamaged metal surface.",
    # # 强调整体结构完整（针对 missing thread 类异常）
    # "a close-up photo of a complete screw with no missing parts.",
    # ],
    'screw': [
        # 简洁的整体正常状态，贴近自然语言
        "a photo of a screw in good condition.",
        # 针对 scratch 类异常，避免 texture/structural 这类词
        "a close-up photo of a screw with no scratches or marks.",
        # 针对 thread 类异常，用可视化的自然描述
        "a close-up photo of a screw with complete and undamaged threading.",
    ],
    # 'tile': [
    #     "a close-up photo of a tile without damage.",
    #     "a close-up photo of a tile without defect.",
    #     "a close-up photo of the tile without damage."
    # ],
    'tile': [
    # 强调瓷砖表面光洁无裂纹（针对 crack / rough）
    "a close-up photo of a tile with smooth and clean surface texture.",
    # 强调颜色均匀无污渍（针对 oil / gray stroke）
    "a close-up photo of a tile with uniform color and no stains or streaks.",
    # 强调边缘完整无粘连（针对 glue）
    "a close-up photo of a tile with intact edges and no adhesive residue.",
    ],
    'toothbrush': [
        "a photo of a unblemished toothbrush for visual inspection.",
        "a photo of the toothbrush without defect.",
        "a close-up photo of the toothbrush without defect."
    ],
    'transistor': [
        "a photo of a transistor without damage for anomaly detection.",
        "a photo of the transistor without damage for anomaly detection.",
        "a photo of the small transistor without damage."
    ],
    'wood': [
        "a cropped photo of the wood surface without defect.",
        "a cropped photo of the wood surface without defect.",
        "a close-up photo of the wood surface without defect."
    ],
    'zipper': [
        "a photo of the unblemished zipper for anomaly detection.",
        "a cropped photo of a zipper without flaw.",
        "a cropped photo of the zipper without flaw."
    ],
}
