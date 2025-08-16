#!/usr/bin/env python3
"""
PLY to Top-View Converter 사용 예제

이 스크립트는 다양한 사용 시나리오를 보여줍니다.
"""

import sys
import os
from pathlib import Path

# 상위 디렉토리의 모듈 import
sys.path.append(str(Path(__file__).parent.parent))
from ply_to_topview import PLYProcessor

def example_single_file():
    """단일 파일 처리 예제"""
    print("=== 단일 파일 처리 예제 ===")
    
    # PLY 파일 경로 (실제 파일로 변경하세요)
    ply_file = "/root/data/datasets/RELLIS-3D/Rellis-3D/00000/ply_frames_Pi3/ply_win10_gap5_step20_s000100_e000145.ply"
    
    if not os.path.exists(ply_file):
        print(f"예제 파일이 존재하지 않습니다: {ply_file}")
        print("실제 PLY 파일 경로로 변경하세요.")
        return
    
    # 프로세서 생성 (지면 평면 시각화 포함)
    processor = PLYProcessor(
        output_dir="example_results",
        visualize_ground_plane=True
    )
    
    # 다양한 해상도로 처리
    resolutions = [0.05, 0.03, 0.02, 0.01]  # 저해상도부터 고해상도까지
    
    success = processor.process_single_ply(ply_file, resolutions)
    
    if success:
        print("✅ 단일 파일 처리 완료!")
    else:
        print("❌ 처리 실패")

def example_custom_parameters():
    """커스텀 매개변수 예제"""
    print("\n=== 커스텀 매개변수 예제 ===")
    
    # 커스텀 프로세서 (설정 예제)
    processor = PLYProcessor(
        output_dir="custom_results",
        visualize_ground_plane=True
    )
    
    # PLY 파일 경로
    ply_file = "/root/data/datasets/RELLIS-3D/Rellis-3D/00000/ply_frames_Pi3/ply_win10_gap5_step20_s000100_e000145.ply"
    
    if not os.path.exists(ply_file):
        print(f"예제 파일이 존재하지 않습니다: {ply_file}")
        return
    
    # PLY 데이터 로드
    points, normals, colors = processor.read_ply_binary(ply_file)
    if points is None:
        return
    
    # 지면 평면 검출
    ground_normal, ground_plane_result = processor.detect_ground_plane(points)
    
    # 커스텀 해상도로 탑뷰 이미지 생성
    custom_resolution = 0.015  # 1.5cm/pixel
    
    print(f"커스텀 해상도로 처리: {custom_resolution}m/pixel")
    
    # 서브샘플링 (메모리 절약)
    if len(points) > 500000:
        import numpy as np
        indices = np.random.choice(len(points), 500000, replace=False)
        points_sub = points[indices]
        colors_sub = colors[indices]
    else:
        points_sub = points
        colors_sub = colors
    
    # 탑뷰 이미지 생성
    image, bounds, size = processor.create_topview_image(
        points_sub, colors_sub, ground_normal, 
        resolution=custom_resolution, 
        blur_sigma=0.5  # 더 세밀한 블러
    )
    
    # 커스텀 출력 경로
    output_path = Path("custom_results") / "custom_topview.png"
    output_path.parent.mkdir(exist_ok=True)
    
    processor.save_topview_image(
        image, bounds, size, ground_normal, str(output_path),
        f"Custom Resolution {custom_resolution}m/pixel"
    )
    
    print("✅ 커스텀 매개변수 처리 완료!")

def example_batch_processing():
    """배치 처리 예제"""
    print("\n=== 배치 처리 예제 ===")
    
    # 예제 디렉토리 (실제 PLY 파일들이 있는 디렉토리로 변경)
    ply_directory = "/root/data/datasets/RELLIS-3D/Rellis-3D/00000/ply_frames_Pi3/"
    
    if not os.path.exists(ply_directory):
        print(f"예제 디렉토리가 존재하지 않습니다: {ply_directory}")
        print("실제 PLY 파일들이 있는 디렉토리로 변경하세요.")
        return
    
    processor = PLYProcessor(
        output_dir="batch_results",
        visualize_ground_plane=False  # 배치 처리시 시각화 비활성화 (빠른 처리)
    )
    
    # 빠른 배치 처리용 해상도
    fast_resolutions = [0.05, 0.03]  # 저해상도와 중해상도만
    
    print(f"디렉토리 배치 처리: {ply_directory}")
    success = processor.process_directory(ply_directory, fast_resolutions)
    
    if success:
        print("✅ 배치 처리 완료!")
    else:
        print("❌ 배치 처리 실패")

def main():
    """메인 함수"""
    print("PLY to Top-View Converter 사용 예제")
    print("=" * 50)
    
    # 예제 1: 단일 파일 처리
    example_single_file()
    
    # 예제 2: 커스텀 매개변수
    example_custom_parameters()
    
    # 예제 3: 배치 처리 (주석 해제하여 사용)
    # example_batch_processing()
    
    print("\n" + "=" * 50)
    print("모든 예제 완료!")
    print("결과 파일들을 확인하세요:")
    print("  - example_results/")
    print("  - custom_results/")
    print("  - batch_results/ (배치 처리 활성화시)")

if __name__ == "__main__":
    main()

