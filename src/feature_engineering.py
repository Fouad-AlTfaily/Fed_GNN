"""
Feature engineering for FedGATSage specialized detectors.
Extracts community-aware features for temporal, content, and behavioral attack detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles the extraction of specialized features for each GAT detector.
    This ensures that the Temporal, Content, and Behavioral models each get the 
    data they need to excel at their specific tasks.
    """
    
    def __init__(self, detector_type='temporal'):
        self.detector_type = detector_type
        self.created_features = []
        
        # We group attacks to ensure the right features are generated for the right problem
        self.temporal_attacks = ['ddos', 'dos', 'scanning']
        self.content_attacks = ['injection', 'xss'] 
        self.behavioral_attacks = ['password', 'backdoor', 'ransomware']
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: takes raw data and adds the specialized columns 
        needed for the current detector type.
        """
        result_df = df.copy()
        
        # First, everyone gets the basics (flow rates, payload sizes)
        result_df = self._add_base_features(result_df)
        
        # Then we add the specialized features
        if self.detector_type == 'temporal':
            result_df = self._add_temporal_features(result_df)
        elif self.detector_type == 'content':
            result_df = self._add_content_features(result_df)
        elif self.detector_type == 'behavioral':
            result_df = self._add_behavioral_features(result_df)
            
        logger.info(f"Engineered {len(self.created_features)} new features for {self.detector_type} detection")
        return result_df
    
    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add base traffic features for all detector types"""
        
        # Flow rate features
        if 'Flow Duration' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['flow_rate'] = (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']) / (df['Flow Duration'] / 1000000 + 1e-6)
            self.created_features.append('flow_rate')
        
        # Payload size features
        if 'TotLen Fwd Pkts' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['avg_payload_fwd'] = df['TotLen Fwd Pkts'] / (df['Tot Fwd Pkts'] + 1e-6)
            df['avg_payload_bwd'] = df['TotLen Bwd Pkts'] / (df['Tot Bwd Pkts'] + 1e-6)
            self.created_features.extend(['avg_payload_fwd', 'avg_payload_bwd'])
        
        # Protocol encoding
        if 'Protocol' in df.columns:
            df['protocol_encoded'] = pd.Categorical(df['Protocol']).codes
            self.created_features.append('protocol_encoded')
            
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal attack-specific features"""
        
        # Inter-arrival time features
        if 'Flow IAT Mean' in df.columns:
            df['iat_variance'] = df['Flow IAT Std'] / (df['Flow IAT Mean'] + 1e-6)
            self.created_features.append('iat_variance')
        
        # Burst detection
        if 'Flow Pkts/s' in df.columns:
            mean_pps = df['Flow Pkts/s'].mean()
            df['burst_ratio'] = df['Flow Pkts/s'] / (mean_pps + 1e-6)
            df['is_burst'] = (df['burst_ratio'] > 2.0).astype(int)
            self.created_features.extend(['burst_ratio', 'is_burst'])
        
        # Flag patterns
        flag_cols = ['SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt']
        if all(col in df.columns for col in flag_cols):
            df['syn_rst_ratio'] = df['SYN Flag Cnt'] / (df['RST Flag Cnt'] + 1e-6)
            df['unusual_flags'] = ((df['SYN Flag Cnt'] > 0) & (df['RST Flag Cnt'] > 0)).astype(int)
            self.created_features.extend(['syn_rst_ratio', 'unusual_flags'])
            
        return df
    
    def _add_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add content attack-specific features"""
        
        # Port analysis
        if 'Dst Port' in df.columns:
            web_ports = [80, 443, 8080, 8443]
            db_ports = [1433, 1521, 3306, 5432]
            
            df['is_web_port'] = df['Dst Port'].isin(web_ports).astype(int)
            df['is_db_port'] = df['Dst Port'].isin(db_ports).astype(int)
            self.created_features.extend(['is_web_port', 'is_db_port'])
        
        # Payload size analysis
        if 'TotLen Fwd Pkts' in df.columns:
            mean_payload = df['TotLen Fwd Pkts'].mean()
            std_payload = df['TotLen Fwd Pkts'].std()
            
            df['unusual_payload'] = (df['TotLen Fwd Pkts'] > mean_payload + 2*std_payload).astype(int)
            df['payload_ratio'] = df['TotLen Fwd Pkts'] / (df['TotLen Bwd Pkts'] + 1e-6)
            self.created_features.extend(['unusual_payload', 'payload_ratio'])
            
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral attack-specific features"""
        
        # Connection pattern analysis
        if 'Src Port' in df.columns and 'Dst Port' in df.columns:
            df['is_ephemeral_src'] = (df['Src Port'] > 1024).astype(int)
            df['targets_system_port'] = (df['Dst Port'] < 1024).astype(int)
            df['port_spread'] = abs(df['Src Port'] - df['Dst Port'])
            self.created_features.extend(['is_ephemeral_src', 'targets_system_port', 'port_spread'])
        
        # Session characteristics
        if 'Flow Duration' in df.columns:
            median_duration = df['Flow Duration'].median()
            df['is_short_session'] = (df['Flow Duration'] < median_duration/10).astype(int)
            df['is_long_session'] = (df['Flow Duration'] > median_duration*10).astype(int)
            self.created_features.extend(['is_short_session', 'is_long_session'])
        
        # Volume analysis
        if 'TotLen Fwd Pkts' in df.columns:
            df['is_low_volume'] = ((df['TotLen Fwd Pkts'] < 100) & (df['Tot Fwd Pkts'] < 5)).astype(int)
            self.created_features.append('is_low_volume')
            
        return df

class CentralityFeatureExtractor:
    """Extract community-aware centrality features"""
    
    def __init__(self):
        self.centrality_cache = {}
    
    def extract_centrality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract centrality features that capture community structure.
        Assumes centrality measures are pre-computed in dataset.
        """
        centrality_cols = [col for col in df.columns if any(measure in col.lower() for measure in [
            'betweenness', 'pagerank', 'degree', 'closeness', 'eigenvector',
            'k_core', 'k_truss', 'modularity'
        ])]
        
        if not centrality_cols:
            logger.warning("No centrality features found in dataset")
            return df
        
        logger.info(f"Found {len(centrality_cols)} centrality features: {centrality_cols}")
        return df



