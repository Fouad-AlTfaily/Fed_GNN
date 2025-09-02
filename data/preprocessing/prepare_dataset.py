"""
Data preparation script for FedGATSage.
Converts raw IoT datasets into the required format with community-aware features.
"""

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from community_detection import CommunityAwareProcessor
from feature_engineering import FeatureEngineer, CentralityFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Preprocess raw IoT datasets for FedGATSage"""
    
    def __init__(self, dataset_name: str, raw_data_path: str, output_dir: str):
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.community_processor = CommunityAwareProcessor()
        
        # Create output directories
        self.detector_dirs = {}
        for detector_type in ['temporal', 'content', 'behavioral']:
            detector_dir = os.path.join(output_dir, f'{detector_type}_detector')
            os.makedirs(detector_dir, exist_ok=True)
            self.detector_dirs[detector_type] = detector_dir
        
        logger.info(f"Initialized preprocessor for {dataset_name}")
    
    def preprocess_full_pipeline(self, num_clients: int = 5, test_split: float = 0.2) -> Dict[str, any]:
        """Run complete preprocessing pipeline"""
        logger.info("Starting full preprocessing pipeline")
        
        # Step 1: Load and clean raw data
        df = self.load_raw_data()
        df = self.clean_data(df)
        
        # Step 2: Add community-aware centrality features
        df = self.add_centrality_features(df)
        
        # Step 3: Create federated data split
        client_data, test_data = self.create_federated_split(df, num_clients, test_split)
        
        # Step 4: Apply detector-specific feature engineering
        self.create_detector_datasets(client_data, test_data)
        
        # Step 5: Save metadata
        self.save_metadata(df, client_data, test_data)
        
        logger.info("Preprocessing pipeline completed")
        return {
            'total_samples': len(df),
            'num_clients': num_clients,
            'test_samples': len(test_data),
            'attack_distribution': df['Attack'].value_counts().to_dict()
        }
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset based on dataset type"""
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        if self.dataset_name.lower() in ['cic_ton_iot', 'cic-ton-iot']:
            return self.load_cic_ton_iot()
        elif self.dataset_name.lower() in ['nf_ton_iot', 'nf-ton-iot']:
            return self.load_nf_ton_iot()
        else:
            # Generic CSV loading
            return pd.read_csv(self.raw_data_path)
    
    def load_cic_ton_iot(self) -> pd.DataFrame:
        """Load and standardize CIC-ToN-IoT dataset"""
        df = pd.read_csv(self.raw_data_path)
        
        # Standardize column names for CIC-ToN-IoT
        column_mapping = {
            'src_ip': 'Src IP',
            'dst_ip': 'Dst IP',
            'src_port': 'Src Port', 
            'dst_port': 'Dst Port',
            'protocol': 'Protocol',
            'flow_duration': 'Flow Duration',
            'tot_fwd_pkts': 'Tot Fwd Pkts',
            'tot_bwd_pkts': 'Tot Bwd Pkts',
            'totlen_fwd_pkts': 'TotLen Fwd Pkts',
            'totlen_bwd_pkts': 'TotLen Bwd Pkts',
            'label': 'Attack'
        }
        
        # Apply mapping if columns exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        logger.info(f"Loaded CIC-ToN-IoT dataset: {len(df)} samples, {len(df.columns)} features")
        return df
    
    def load_nf_ton_iot(self) -> pd.DataFrame:
        """Load and standardize NF-ToN-IoT dataset"""  
        df = pd.read_csv(self.raw_data_path)
        
        # NF-ToN-IoT typically has different column names
        column_mapping = {
            'IPV4_SRC_ADDR': 'Src IP',
            'IPV4_DST_ADDR': 'Dst IP', 
            'L4_SRC_PORT': 'Src Port',
            'L4_DST_PORT': 'Dst Port',
            'PROTOCOL': 'Protocol',
            'Label': 'Attack'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        logger.info(f"Loaded NF-ToN-IoT dataset: {len(df)} samples, {len(df.columns)} features")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        logger.info("Cleaning dataset")
        
        initial_size = len(df)
        
        # Remove rows with missing critical fields
        required_cols = ['Src IP', 'Dst IP', 'Attack']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Dataset must contain columns: {required_cols}")
        
        # Drop rows with missing values in critical columns
        df = df.dropna(subset=required_cols)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Clean attack labels
        df['Attack'] = df['Attack'].astype(str).str.strip().str.lower()
        
        # Map common attack label variations
        attack_mapping = {
            'benign': 'Benign',
            'normal': 'Benign',
            'ddos': 'ddos',
            'dos': 'dos', 
            'injection': 'injection',
            'xss': 'xss',
            'password': 'password',
            'scanning': 'scanning',
            'backdoor': 'backdoor',
            'ransomware': 'ransomware',
            'mitm': 'mitm'
        }
        
        df['Attack'] = df['Attack'].map(attack_mapping).fillna(df['Attack'])
        
        logger.info(f"Cleaned dataset: {len(df)} samples (removed {initial_size - len(df)})")
        logger.info(f"Attack distribution: {df['Attack'].value_counts().to_dict()}")
        
        return df
    
    def add_centrality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add community-aware centrality features"""
        logger.info("Computing centrality features")
        
        try:
            # Create network graph
            G = nx.Graph()
            for _, row in df.iterrows():
                src, dst = row['Src IP'], row['Dst IP']
                if G.has_edge(src, dst):
                    G[src][dst]['weight'] += 1
                else:
                    G.add_edge(src, dst, weight=1)
            
            logger.info(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Compute centrality measures
            centralities = {}
            
            # Degree centrality
            try:
                centralities['degree'] = nx.degree_centrality(G)
            except:
                logger.warning("Could not compute degree centrality")
            
            # Betweenness centrality (sample for large graphs)
            try:
                if G.number_of_nodes() > 1000:
                    k = min(100, G.number_of_nodes())
                    centralities['betweenness'] = nx.betweenness_centrality(G, k=k)
                else:
                    centralities['betweenness'] = nx.betweenness_centrality(G)
            except:
                logger.warning("Could not compute betweenness centrality")
            
            # PageRank
            try:
                centralities['pagerank'] = nx.pagerank(G, max_iter=50)
            except:
                logger.warning("Could not compute PageRank")
            
            # Closeness centrality (sample for large graphs)
            try:
                if G.number_of_nodes() > 1000:
                    # Sample nodes for closeness computation
                    sample_nodes = list(G.nodes())[:100]
                    closeness_sample = nx.closeness_centrality(G.subgraph(sample_nodes))
                    centralities['closeness'] = {node: closeness_sample.get(node, 0) for node in G.nodes()}
                else:
                    centralities['closeness'] = nx.closeness_centrality(G)
            except:
                logger.warning("Could not compute closeness centrality")
            
            # Add centrality features to dataframe
            for measure, values in centralities.items():
                df[f'src_{measure}'] = df['Src IP'].map(values).fillna(0)
                df[f'dst_{measure}'] = df['Dst IP'].map(values).fillna(0)
                df[f'flow_{measure}_ratio'] = df[f'dst_{measure}'] / (df[f'src_{measure}'] + 1e-6)
            
            logger.info(f"Added {len(centralities)} centrality measures")
            
        except Exception as e:
            logger.error(f"Error computing centrality features: {e}")
            # Add dummy centrality features if computation fails
            dummy_centralities = ['degree', 'betweenness', 'pagerank', 'closeness']
            for measure in dummy_centralities:
                df[f'src_{measure}'] = 0.0
                df[f'dst_{measure}'] = 0.0
                df[f'flow_{measure}_ratio'] = 1.0
        
        return df
    
    def create_federated_split(self, df: pd.DataFrame, num_clients: int, 
                             test_split: float) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """Create federated data split with balanced attack distribution"""
        logger.info(f"Creating federated split: {num_clients} clients, {test_split} test split")
        
        # First split: train/test
        train_data, test_data = train_test_split(
            df, test_size=test_split, random_state=42, 
            stratify=df['Attack'], shuffle=True
        )
        
        logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Split training data among clients
        client_data = []
        attack_types = train_data['Attack'].unique()
        
        for client_id in range(num_clients):
            client_samples = []
            
            # For each attack type, assign samples to this client
            for attack in attack_types:
                attack_data = train_data[train_data['Attack'] == attack]
                
                # Calculate client's share
                start_idx = (len(attack_data) * client_id) // num_clients
                end_idx = (len(attack_data) * (client_id + 1)) // num_clients
                
                client_attack_data = attack_data.iloc[start_idx:end_idx]
                client_samples.append(client_attack_data)
            
            # Combine all attack types for this client
            client_df = pd.concat(client_samples, ignore_index=True)
            client_df = client_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            client_data.append(client_df)
            
            logger.info(f"Client {client_id + 1}: {len(client_df)} samples")
        
        return client_data, test_data
    
    def create_detector_datasets(self, client_data: List[pd.DataFrame], test_data: pd.DataFrame):
        """Create specialized datasets for each detector type"""
        logger.info("Creating detector-specific datasets")
        
        for detector_type in ['temporal', 'content', 'behavioral']:
            detector_dir = self.detector_dirs[detector_type]
            feature_engineer = FeatureEngineer(detector_type)
            
            # Process test data
            test_enhanced = feature_engineer.extract_features(test_data.copy())
            test_enhanced.to_csv(os.path.join(detector_dir, 'test.csv'), index=False)
            
            # Process client data
            for client_id, client_df in enumerate(client_data):
                client_enhanced = feature_engineer.extract_features(client_df.copy())
                client_path = os.path.join(detector_dir, f'client_{client_id + 1}.csv')
                client_enhanced.to_csv(client_path, index=False)
            
            logger.info(f"Created {detector_type} detector datasets")
    
    def save_metadata(self, original_df: pd.DataFrame, client_data: List[pd.DataFrame], 
                     test_data: pd.DataFrame):
        """Save dataset metadata"""
        metadata = {
            'dataset_name': self.dataset_name,
            'original_samples': len(original_df),
            'test_samples': len(test_data),
            'num_clients': len(client_data),
            'client_sizes': [len(df) for df in client_data],
            'attack_distribution': original_df['Attack'].value_counts().to_dict(),
            'test_attack_distribution': test_data['Attack'].value_counts().to_dict(),
            'feature_columns': list(original_df.columns),
            'centrality_features': [col for col in original_df.columns 
                                  if any(measure in col for measure in ['degree', 'betweenness', 'pagerank', 'closeness'])]
        }
        
        # Save to all detector directories
        for detector_dir in self.detector_dirs.values():
            metadata_path = os.path.join(detector_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
        
        logger.info("Metadata saved to all detector directories")

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Prepare dataset for FedGATSage')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cic_ton_iot', 'nf_ton_iot'], 
                       help='Dataset name')
    parser.add_argument('--raw_data_path', type=str, required=True,
                       help='Path to raw dataset CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='Number of federated clients')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Test set split ratio')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(
        dataset_name=args.dataset,
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir
    )
    
    # Run preprocessing
    try:
        results = preprocessor.preprocess_full_pipeline(
            num_clients=args.num_clients,
            test_split=args.test_split
        )
        
        print("\n" + "="*50)
        print("DATA PREPROCESSING COMPLETED")
        print("="*50)
        print(f"Total samples processed: {results['total_samples']}")
        print(f"Clients created: {results['num_clients']}")
        print(f"Test samples: {results['test_samples']}")
        print(f"Attack distribution:")
        for attack, count in results['attack_distribution'].items():
            print(f"  {attack}: {count}")
        print(f"\nProcessed data saved to: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()