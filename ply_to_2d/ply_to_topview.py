#!/usr/bin/env python3
"""
PLY to Top-View Image Converter

PLY 포인트 클라우드 파일(들)을 입력으로 받아 2D top-down RGB 이미지를 생성하는 스크립트

Features:
- 단일 PLY 파일 또는 디렉토리 처리
- 자동 지면 평면 검출 (RANSAC)
- 다양한 해상도 지원
- 지면 평면 시각화 옵션
- 배치 처리 지원

Date: 2024-08-16
"""

import os
import sys
import argparse
import glob
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.ndimage import gaussian_filter, median_filter
from pathlib import Path
import json
import time

class PLYProcessor:
    """PLY 파일 처리 및 2D 탑뷰 이미지 생성 클래스"""
    
    def __init__(self, output_dir="results", visualize_ground_plane=False, advanced_postprocessing=False, use_all_points=False, preserve_orientation=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.visualize_ground_plane = visualize_ground_plane
        self.advanced_postprocessing = advanced_postprocessing
        self.use_all_points = use_all_points
        self.preserve_orientation = preserve_orientation
        
    def read_ply_binary(self, filepath):
        """Binary PLY 파일을 읽어서 포인트 클라우드 데이터를 반환"""
        print(f"PLY 파일 로딩: {filepath}")
        
        points = []
        normals = []
        colors = []
        
        try:
            with open(filepath, 'rb') as f:
                # 헤더 읽기
                line = f.readline().decode('ascii').strip()
                if line != 'ply':
                    raise ValueError("Not a PLY file")
                
                vertex_count = 0
                has_normals = False
                has_colors = False
                
                while True:
                    line = f.readline().decode('ascii').strip()
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('property float nx'):
                        has_normals = True
                    elif line.startswith('property uchar red'):
                        has_colors = True
                    elif line == 'end_header':
                        break
                
                print(f"  포인트 개수: {vertex_count:,}")
                print(f"  법선 벡터: {'있음' if has_normals else '없음'}")
                print(f"  색상 정보: {'있음' if has_colors else '없음'}")
                
                # Binary 데이터 읽기
                bytes_per_vertex = 4 * 3  # x, y, z
                if has_normals:
                    bytes_per_vertex += 4 * 3  # nx, ny, nz
                if has_colors:
                    bytes_per_vertex += 3  # r, g, b
                
                for i in range(vertex_count):
                    data = f.read(bytes_per_vertex)
                    if len(data) < bytes_per_vertex:
                        break
                    
                    if has_normals and has_colors:
                        # x, y, z, nx, ny, nz, r, g, b
                        coords = struct.unpack('<6f3B', data)
                        points.append([coords[0], coords[1], coords[2]])
                        normals.append([coords[3], coords[4], coords[5]])
                        colors.append([coords[6]/255.0, coords[7]/255.0, coords[8]/255.0])
                    elif has_colors:
                        # x, y, z, r, g, b
                        coords = struct.unpack('<3f3B', data)
                        points.append([coords[0], coords[1], coords[2]])
                        colors.append([coords[3]/255.0, coords[4]/255.0, coords[5]/255.0])
                    else:
                        # x, y, z만
                        coords = struct.unpack('<3f', data)
                        points.append([coords[0], coords[1], coords[2]])
                        colors.append([0.5, 0.5, 0.5])  # 기본 회색
                    
                    if i % 200000 == 0:
                        print(f"  진행: {i:,}/{vertex_count:,} ({i/vertex_count*100:.1f}%)")
                
                if not has_normals:
                    normals = np.zeros_like(points)
                    
        except Exception as e:
            print(f"PLY 파일 읽기 오류: {e}")
            return None, None, None
            
        return np.array(points), np.array(normals), np.array(colors)
    
    def fit_plane_ransac(self, points, num_iterations=2000, threshold=0.05, min_points=1000):
        """RANSAC을 사용해서 평면을 피팅"""
        print(f"RANSAC 평면 피팅 시작...")
        print(f"  반복 횟수: {num_iterations}")
        print(f"  임계값: {threshold}m")
        
        best_plane = None
        best_inliers = 0
        best_inlier_mask = None
        
        for i in range(num_iterations):
            # 3개의 랜덤 점 선택
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]
            
            # 평면의 법선 벡터 계산
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            
            # 법선 벡터 정규화
            normal_length = np.linalg.norm(normal)
            if normal_length < 1e-6:
                continue
            normal = normal / normal_length
            
            # 평면 방정식: ax + by + cz + d = 0
            d = -np.dot(normal, sample_points[0])
            
            # 모든 점에서 평면까지의 거리 계산
            distances = np.abs(np.dot(points, normal) + d)
            
            # 인라이어 찾기
            inlier_mask = distances < threshold
            num_inliers = np.sum(inlier_mask)
            
            # 최고 결과 업데이트
            if num_inliers > best_inliers and num_inliers > min_points:
                best_inliers = num_inliers
                best_plane = (normal, d)
                best_inlier_mask = inlier_mask
            
            if i % 200 == 0:
                print(f"  반복 {i}: 최대 인라이어 = {best_inliers:,}")
        
        if best_plane is None:
            print("  경고: 적절한 평면을 찾지 못했습니다!")
            return None, None
            
        normal, d = best_plane
        
        # 법선 벡터 원본 유지 (실제 지면 방향 보존)
        print(f"  검출된 법선 벡터: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
        print("  원본 법선 벡터 방향 유지 (실제 지면 방향)")
            
        print(f"  완료: 인라이어 {best_inliers:,}개 ({best_inliers/len(points)*100:.1f}%)")
        print(f"  법선 벡터: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
        
        return best_plane, best_inlier_mask
    
    def detect_ground_plane_with_normals(self, points, normals):
        """법선 벡터 정보를 활용한 향상된 지면 평면 검출"""
        print("향상된 Ground Plane Detection (법선 벡터 활용)...")
        
        # 1. 법선 벡터가 있는지 확인
        has_valid_normals = np.any(np.linalg.norm(normals, axis=1) > 0.1)
        
        if has_valid_normals:
            print("  법선 벡터 정보 활용 중...")
            
            # 2. 지면을 향하는 법선 벡터 찾기 (수직에 가까운 벡터들)
            normal_lengths = np.linalg.norm(normals, axis=1)
            valid_normals_mask = normal_lengths > 0.1
            
            valid_normals = normals[valid_normals_mask]
            valid_points = points[valid_normals_mask]
            
            # 3. Z축과의 각도 계산 (지면 법선은 수직에 가까워야 함)
            z_axis = np.array([0, 0, 1])
            angles_to_z = np.abs(np.dot(valid_normals, z_axis))  # 코사인 값
            
            # 4. 수직에 가까운 법선들 선택 (cos(30°) ≈ 0.866 이상)
            vertical_mask = angles_to_z > 0.5
            vertical_normals = valid_normals[vertical_mask]
            vertical_points = valid_points[vertical_mask]
            
            if len(vertical_normals) > 100:
                print(f"  수직 법선 벡터 {len(vertical_normals):,}개 발견")
                
                # 5. 법선 벡터들의 평균으로 지면 방향 추정
                mean_normal = np.mean(vertical_normals, axis=0)
                mean_normal = mean_normal / np.linalg.norm(mean_normal)
                
                # 6. 지면 높이 추정 (수직 법선을 가진 점들의 평균)
                mean_height = np.mean(vertical_points[:, 2])
                d = -mean_height * mean_normal[2]
                
                print(f"  법선 벡터 기반 지면 검출 성공")
                print(f"  평균 법선 벡터: [{mean_normal[0]:.4f}, {mean_normal[1]:.4f}, {mean_normal[2]:.4f}]")
                
                return mean_normal, {
                    'normal': mean_normal,
                    'd': d,
                    'method': 'normal_vectors',
                    'vertical_points_count': len(vertical_normals),
                    'confidence': min(len(vertical_normals) / 1000, 1.0)
                }
        
        # 7. 법선 벡터 방법이 실패하면 기존 RANSAC 방법 사용
        print("  법선 벡터 불충분 → RANSAC 방법으로 fallback")
        return self.detect_ground_plane_ransac(points)
    
    def detect_ground_plane_ransac(self, points):
        """기존 RANSAC 기반 지면 평면 검출"""
        # Z 값이 낮은 점들만 사용 (지면 후보)
        z_percentile_20 = np.percentile(points[:, 2], 20)
        ground_candidates = points[points[:, 2] < z_percentile_20]
        print(f"  Ground Candidates: {len(ground_candidates):,} (20% Z-value)")
        
        # RANSAC으로 평면 피팅
        plane_result = self.fit_plane_ransac(ground_candidates)
        
        if plane_result[0] is None:
            print("  Warning: Failed to detect ground plane, using default value")   
            return np.array([0, 0, 1]), None  # 기본 Z축
            
        normal, d = plane_result[0]
        inlier_mask = plane_result[1]
        
        return normal, {
            'normal': normal,
            'd': d,
            'method': 'ransac',
            'ground_candidates': ground_candidates,
            'inliers': ground_candidates[inlier_mask],
            'inlier_ratio': np.sum(inlier_mask) / len(ground_candidates)
        }
    
    def detect_ground_plane(self, points, normals=None):
        """통합 지면 평면 검출 (법선 벡터 우선, RANSAC fallback)"""
        if normals is not None:
            return self.detect_ground_plane_with_normals(points, normals)
        else:
            return self.detect_ground_plane_ransac(points)
    
    def create_rotation_matrix(self, normal_vector):
        """Ground 법선 벡터를 Z축으로 회전시키는 회전 행렬을 생성"""
        target = np.array([0, 0, 1])
        normal = normal_vector / np.linalg.norm(normal_vector)
        
        if np.allclose(normal, target):
            return np.eye(3)
        
        rotation_axis = np.cross(normal, target)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        cos_angle = np.dot(normal, target)
        sin_angle = np.linalg.norm(np.cross(normal, target))
        angle = np.arctan2(sin_angle, cos_angle)
        
        # 로드리게스 회전 공식
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R
    
    def apply_advanced_postprocessing(self, image, resolution):
        """고급 이미지 후처리 (옵션)"""
        if not self.advanced_postprocessing:
            return image
            
        print("  고급 후처리 적용 중...")
        enhanced = image.copy()
        
        # 1. 노이즈 감소 (메디안 필터)
        for c in range(3):
            enhanced[:, :, c] = median_filter(enhanced[:, :, c], size=3)
        
        # 2. 적응적 히스토그램 균등화 (CLAHE)
        if np.max(enhanced) > 0:
            enhanced_uint8 = (enhanced * 255).astype(np.uint8)
            lab = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2LAB)
            
            # L 채널에 CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            enhanced_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            enhanced = enhanced_uint8.astype(np.float64) / 255.0
        
        # 3. 언샤프 마스킹 (선명도 향상)
        if resolution <= 0.005:  # 5mm/pixel 이하에서만 적용
            blurred = gaussian_filter(enhanced, sigma=1.0)
            enhanced = enhanced + 0.5 * (enhanced - blurred)
            enhanced = np.clip(enhanced, 0, 1)
        
        # 4. 색상 보정
        # enhanced = np.clip(enhanced * 1.05, 0, 1)  # 약간의 밝기 증가
        
        return enhanced
    
    def create_topview_image(self, points, colors, ground_normal, resolution=0.03, blur_sigma=0.8):
        """탑뷰 이미지 생성"""
        print(f"탑뷰 이미지 생성 중... (해상도: {resolution}m/pixel)")
        
        # 좌표계 변환 (선택적)
        if self.preserve_orientation:
            print("  원본 좌표계 보존 모드")
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            z_coords = points[:, 2]
        else:
            print("  지면 평면 기준 좌표계 변환")
            R = self.create_rotation_matrix(ground_normal)
            rotated_points = np.dot(points, R.T)
            x_coords = rotated_points[:, 0]
            y_coords = rotated_points[:, 1]
            z_coords = rotated_points[:, 2]
        
        # 좌표 범위 계산
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        print(f"  좌표 범위: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
        
        # 이미지 크기 계산
        width = int((x_max - x_min) / resolution) + 1
        height = int((y_max - y_min) / resolution) + 1
        
        print(f"  이미지 크기: {width} x {height}")
        
        # 이미지 버퍼들
        image_rgb = np.zeros((height, width, 3), dtype=np.float64)
        depth_buffer = np.full((height, width), -np.inf)
        
        # 포인트를 이미지에 투영
        pixel_x = ((x_coords - x_min) / resolution).astype(int)
        pixel_y = ((y_coords - y_min) / resolution).astype(int)
        
        # 범위 내 포인트만 선택
        valid_mask = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
        
        valid_x = pixel_x[valid_mask]
        valid_y = pixel_y[valid_mask]
        valid_z = z_coords[valid_mask]
        valid_colors = colors[valid_mask]
        
        print(f"  유효한 포인트: {len(valid_x):,} / {len(points):,}")
        
        # 각 픽셀에 대해 가장 높은 점의 색상 사용
        for i in range(len(valid_x)):
            px, py = valid_x[i], valid_y[i]
            z = valid_z[i]
            color = valid_colors[i]
            
            # Y 좌표 뒤집기 (이미지 좌표계)
            img_y = height - 1 - py
            
            # 깊이 버퍼 확인
            if z > depth_buffer[img_y, px]:
                depth_buffer[img_y, px] = z
                image_rgb[img_y, px] = color
            
            if i % 100000 == 0:
                print(f"    투영 진행: {i:,}/{len(valid_x):,}")
        
        # 빈 픽셀 보간
        if blur_sigma > 0:
            print("  빈 픽셀 보간 중...")
            image_count = (np.sum(image_rgb, axis=2) > 0).astype(float)
            
            for c in range(3):  # R, G, B 채널별로
                channel = image_rgb[:, :, c]
                mask = image_count > 0
                
                if np.sum(mask) > 0:
                    blurred = gaussian_filter(channel, sigma=blur_sigma)
                    channel[~mask] = blurred[~mask]
                    image_rgb[:, :, c] = channel
        
        # 이미지 후처리
        print("  이미지 후처리 중...")
        
        # 기본 후처리 (밝기, 대비, 채도)
        image_enhanced = np.clip(image_rgb * 1.1, 0, 1)
        image_enhanced = np.clip((image_enhanced - 0.5) * 1.3 + 0.5, 0, 1)
        
        # 채도 향상
        if np.max(image_enhanced) > 0:
            hsv = cv2.cvtColor((image_enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
            image_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
        
        # 고급 후처리 (옵션)
        image_enhanced = self.apply_advanced_postprocessing(image_enhanced, resolution)
        
        return image_enhanced, (x_min, x_max, y_min, y_max), (width, height)
    
    def save_topview_image(self, image, bounds, size, ground_normal, output_path, description=""):
        """탑뷰 이미지 저장"""
        x_min, x_max, y_min, y_max = bounds
        width, height = size
        
        # 1. 깨끗한 이미지 (OpenCV) - Y축 뒤집어서 올바른 방향으로
        image_flipped = np.flipud(image)  # Y축 뒤집기
        image_uint8 = (np.clip(image_flipped, 0, 1) * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        clean_path = output_path.replace('.png', '_clean.png')
        cv2.imwrite(clean_path, image_bgr)
        
        # 2. 축과 정보가 포함된 이미지 (Matplotlib)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        im = ax.imshow(image, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                       interpolation='bilinear')
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f'Top-View RGB Projection{": " + description if description else ""}\n'
                    f'Normal Vector: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]\n'
                    f'Resolution: {(x_max-x_min)/width:.4f} m/pixel | Size: {width} x {height}', 
                    fontsize=12)
        
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # 스케일 바
        scalebar_length = 1.0
        scalebar_x = x_min + 0.05 * (x_max - x_min)
        scalebar_y = y_min + 0.05 * (y_max - y_min)
        ax.plot([scalebar_x, scalebar_x + scalebar_length], 
               [scalebar_y, scalebar_y], 'w-', linewidth=3)
        ax.text(scalebar_x + scalebar_length/2, scalebar_y + 0.02 * (y_max - y_min), 
               '1m', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  저장됨: {clean_path}")
        print(f"  저장됨: {output_path}")
        
        return clean_path, output_path
    
    def visualize_ground_plane_detection(self, points, ground_plane_result, output_path):
        """지면 평면 검출 결과 시각화"""
        if ground_plane_result is None:
            return
            
        print("지면 평면 검출 시각화 중...")
        
        normal = ground_plane_result['normal']
        inliers = ground_plane_result['inliers']
        ground_candidates = ground_plane_result['ground_candidates']
        
        # 서브샘플링 (시각화 성능)
        max_points = 15000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points_vis = points[indices]
        else:
            points_vis = points
        
        fig = plt.figure(figsize=(18, 6))
        
        # 1. 전체 포인트 클라우드
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                             c=points_vis[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('전체 포인트 클라우드')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
        
        # 2. 지면 검출 과정
        ax2 = fig.add_subplot(132, projection='3d')
        
        # 지면 후보 (회색)
        if len(ground_candidates) > 5000:
            indices = np.random.choice(len(ground_candidates), 5000, replace=False)
            candidates_vis = ground_candidates[indices]
        else:
            candidates_vis = ground_candidates
            
        ax2.scatter(candidates_vis[:, 0], candidates_vis[:, 1], candidates_vis[:, 2],
                   c='gray', s=1, alpha=0.3, label='지면 후보')
        
        # 인라이어 (빨간색)
        if len(inliers) > 3000:
            indices = np.random.choice(len(inliers), 3000, replace=False)
            inliers_vis = inliers[indices]
        else:
            inliers_vis = inliers
            
        ax2.scatter(inliers_vis[:, 0], inliers_vis[:, 1], inliers_vis[:, 2],
                   c='red', s=2, alpha=0.8, label='지면 평면')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('지면 평면 검출')
        ax2.legend()
        
        # 3. 법선 벡터
        ax3 = fig.add_subplot(133, projection='3d')
        
        # 지면 중심점
        center = np.mean(inliers, axis=0)
        
        # 인라이어
        ax3.scatter(inliers_vis[:, 0], inliers_vis[:, 1], inliers_vis[:, 2],
                   c='red', s=1, alpha=0.5, label='지면')
        
        # 법선 벡터
        ax3.quiver(center[0], center[1], center[2],
                  normal[0], normal[1], normal[2],
                  length=2.0, color='blue', arrow_length_ratio=0.1, linewidth=3,
                  label=f'법선 벡터')
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title(f'지면 법선 벡터\n[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  지면 평면 시각화 저장: {output_path}")
    
    def process_single_ply(self, ply_path, resolutions=[0.002, 0.005, 0.01]):
        """단일 PLY 파일 처리"""
        ply_path = Path(ply_path)
        print(f"\n{'='*60}")
        print(f"처리 중: {ply_path.name}")
        print(f"{'='*60}")
        
        # PLY 파일 읽기
        points, normals, colors = self.read_ply_binary(ply_path)
        if points is None:
            return False
            
        # 지면 평면 검출 (법선 벡터 정보 활용)
        ground_normal, ground_plane_result = self.detect_ground_plane(points, normals)
        
        # 출력 디렉토리 생성
        output_subdir = self.output_dir / ply_path.stem
        output_subdir.mkdir(exist_ok=True)
        
        # 지면 평면 시각화 (옵션)
        if self.visualize_ground_plane and ground_plane_result is not None:
            ground_viz_path = output_subdir / f"{ply_path.stem}_ground_plane.png"
            self.visualize_ground_plane_detection(points, ground_plane_result, ground_viz_path)
        
        # 다양한 해상도로 탑뷰 이미지 생성
        results = {}
        for resolution in resolutions:
            # 해상도를 파일명에 적합한 형태로 변환
            if resolution >= 1:
                res_name = f"{resolution:.0f}p000"
            elif resolution >= 0.1:
                res_name = f"{resolution:.1f}".replace('.', 'p')
            elif resolution >= 0.01:
                res_name = f"{resolution:.2f}".replace('.', 'p')
            elif resolution >= 0.001:
                res_name = f"{resolution:.3f}".replace('.', 'p')
            else:
                res_name = f"{resolution:.4f}".replace('.', 'p')
            
            print(f"\n--- 해상도 {resolution}m/pixel ---")
            
            # 서브샘플링 (--use-all-points 옵션으로 비활성화 가능)
            if self.use_all_points:
                print(f"모든 포인트 사용: {len(points):,}개")
                points_sub = points
                colors_sub = colors
            elif resolution <= 0.001 and len(points) > 1500000:
                print(f"서브샘플링: 1,500,000 / {len(points):,}")
                indices = np.random.choice(len(points), 1500000, replace=False)
                points_sub = points[indices]
                colors_sub = colors[indices]
            elif resolution > 0.002 and resolution <= 0.02 and len(points) > 1000000:
                print(f"서브샘플링: 1,000,000 / {len(points):,}")
                indices = np.random.choice(len(points), 1000000, replace=False)
                points_sub = points[indices]
                colors_sub = colors[indices]
            else:
                points_sub = points
                colors_sub = colors
            
            # 탑뷰 이미지 생성 (해상도에 따라 블러 강도 조정)
            if resolution <= 0.001:
                blur_sigma = 1.5  # 극초고해상도: 강한 블러
            elif resolution <= 0.002:
                blur_sigma = 1.2  # 초고해상도: 중간 블러  
            else:
                blur_sigma = 0.8  # 일반 해상도: 기본 블러
                
            image, bounds, size = self.create_topview_image(
                points_sub, colors_sub, ground_normal, 
                resolution=resolution, blur_sigma=blur_sigma
            )
            
            # 이미지 저장
            output_path = output_subdir / f"{ply_path.stem}_topview_{res_name}m.png"
            clean_path, full_path = self.save_topview_image(
                image, bounds, size, ground_normal, str(output_path),
                f"{resolution}m/pixel"
            )
            
            results[resolution] = {
                'clean_path': clean_path,
                'full_path': full_path,
                'bounds': bounds,
                'size': size,
                'resolution': resolution
            }
        
        # 지면 법선 벡터 원본 유지 (실제 좌표계 보존)
            
        # 메타데이터 저장
        metadata = {
            'input_file': str(ply_path),
            'total_points': len(points),
            'ground_normal': ground_normal.tolist(),
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': {res: {k: v for k, v in info.items() if k != 'image'} 
                       for res, info in results.items()}
        }
        
        if ground_plane_result is not None:
            metadata['ground_plane'] = {
                'inlier_ratio': ground_plane_result['inlier_ratio'],
                'd_coefficient': ground_plane_result['d']
            }
        
        metadata_path = output_subdir / f"{ply_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n메타데이터 저장: {metadata_path}")
        print(f"처리 완료: {ply_path.name}")
        
        return True
    
    def process_directory(self, directory_path, resolutions=[0.002, 0.005, 0.01]):
        """디렉토리 내 모든 PLY 파일 배치 처리"""
        directory_path = Path(directory_path)
        ply_files = list(directory_path.glob("*.ply"))
        
        if not ply_files:
            print(f"디렉토리에서 PLY 파일을 찾을 수 없습니다: {directory_path}")
            return False
        
        print(f"\n발견된 PLY 파일: {len(ply_files)}개")
        for i, ply_file in enumerate(ply_files):
            print(f"  {i+1}. {ply_file.name}")
        
        success_count = 0
        for i, ply_file in enumerate(ply_files):
            print(f"\n진행: {i+1}/{len(ply_files)}")
            if self.process_single_ply(ply_file, resolutions):
                success_count += 1
        
        print(f"\n배치 처리 완료: {success_count}/{len(ply_files)} 성공")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="PLY 포인트 클라우드를 2D top-down RGB 이미지로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 파일 처리
  python ply_to_topview.py input.ply
  
  # 디렉토리 배치 처리
  python ply_to_topview.py /path/to/ply/files/
  
  # 특정 해상도로 처리
  python ply_to_topview.py input.ply --resolutions 0.001 0.002 0.005
  
  # 지면 평면 시각화 포함
  python ply_to_topview.py input.ply --visualize-ground-plane
  
  # 고급 후처리 적용
  python ply_to_topview.py input.ply --advanced-postprocessing
  
  # 출력 디렉토리 지정
  python ply_to_topview.py input.ply --output results/
        """
    )
    
    parser.add_argument('input', help='입력 PLY 파일 또는 디렉토리')
    parser.add_argument('--output', '-o', default='results', 
                       help='출력 디렉토리 (기본값: results)')
    parser.add_argument('--resolutions', '-r', nargs='+', type=float, 
                       default=[0.002, 0.005, 0.01],
                       help='해상도 리스트 (m/pixel, 기본값: 0.002 0.005 0.01)')
    parser.add_argument('--visualize-ground-plane', '-v', action='store_true',
                       help='지면 평면 검출 과정 시각화')
    parser.add_argument('--advanced-postprocessing', '-a', action='store_true',
                       help='고급 후처리 적용 (노이즈 감소, CLAHE, 언샤프 마스킹)')
    parser.add_argument('--use-all-points', '-u', action='store_true',
                       help='모든 포인트 사용 (서브샘플링 비활성화) - 메모리 사용량 증가')
    parser.add_argument('--preserve-orientation', '-p', action='store_true',
                       help='원본 좌표계 방향 보존 (지면 평면 변환 비활성화)')
    
    args = parser.parse_args()
    
    # 입력 경로 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"오류: 입력 경로가 존재하지 않습니다: {input_path}")
        sys.exit(1)
    
    # 프로세서 생성
    processor = PLYProcessor(
        output_dir=args.output,
        visualize_ground_plane=args.visualize_ground_plane,
        advanced_postprocessing=args.advanced_postprocessing,
        use_all_points=args.use_all_points,
        preserve_orientation=args.preserve_orientation
    )
    
    print("PLY to Top-View Image Converter")
    print(f"입력: {input_path}")
    print(f"출력: {args.output}")
    print(f"해상도: {args.resolutions}")
    print(f"지면 평면 시각화: {'예' if args.visualize_ground_plane else '아니오'}")
    print(f"고급 후처리: {'예' if args.advanced_postprocessing else '아니오'}")
    print(f"모든 포인트 사용: {'예' if args.use_all_points else '아니오'}")
    print(f"원본 방향 보존: {'예' if args.preserve_orientation else '아니오'}")
    
    # 처리 시작
    start_time = time.time()
    
    if input_path.is_file():
        success = processor.process_single_ply(input_path, args.resolutions)
    else:
        success = processor.process_directory(input_path, args.resolutions)
    
    elapsed_time = time.time() - start_time
    
    if success:
        print(f"\n{'='*60}")
        print(f"전체 처리 완료! (소요 시간: {elapsed_time:.1f}초)")
        print(f"결과는 '{args.output}' 디렉토리에 저장되었습니다.")
        print(f"{'='*60}")
    else:
        print("처리 중 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()
