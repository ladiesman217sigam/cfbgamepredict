import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cfbd
from cfbd.rest import ApiException
from difflib import get_close_matches

# Import functions from heatmap.py
from heatmap import configuration, get_team_records, get_team_stats, get_advanced_stats

def get_team_ratings(year=2025):
    """Get team ratings and strength of schedule"""
    with cfbd.ApiClient(configuration) as api_client:
        ratings_api = cfbd.RatingsApi(api_client)
        
        ratings_data = []
        try:
            # Try to get SP+ ratings which include strength of schedule
            ratings = ratings_api.get_sp(year=year)
            
            for rating in ratings:
                ratings_data.append({
                    'team': rating.team if hasattr(rating, 'team') else '',
                    'sp_rating': getattr(rating, 'rating', 0),
                    'sp_offense': getattr(rating, 'offense', 0) if hasattr(rating, 'offense') else 0,
                    'sp_defense': getattr(rating, 'defense', 0) if hasattr(rating, 'defense') else 0,
                    'sp_special_teams': getattr(rating, 'specialTeams', 0) if hasattr(rating, 'specialTeams') else 0,
                })
        except:
            # SP+ ratings might not be available, try FPI ratings
            try:
                ratings = ratings_api.get_fpi(year=year)
                for rating in ratings:
                    ratings_data.append({
                        'team': rating.team if hasattr(rating, 'team') else '',
                        'fpi_rating': getattr(rating, 'fpi', 0),
                        'strength_of_schedule': getattr(rating, 'strengthOfSchedule', 0) if hasattr(rating, 'strengthOfSchedule') else 0,
                    })
            except:
                # Try SRS (Simple Rating System) which includes strength of schedule
                try:
                    ratings = ratings_api.get_srs(year=year)
                    for rating in ratings:
                        ratings_data.append({
                            'team': rating.team if hasattr(rating, 'team') else '',
                            'srs_rating': getattr(rating, 'srs', 0),
                            'sos': getattr(rating, 'sos', 0) if hasattr(rating, 'sos') else 0,  # Strength of Schedule
                        })
                except:
                    pass
        
        return pd.DataFrame(ratings_data)

def get_all_teams(year=2025):
    """Get list of all available team names"""
    records_df = get_team_records(year)
    if not records_df.empty:
        return sorted(records_df['team'].tolist())
    return []

def find_team_name(user_input, available_teams):
    """Find the best matching team name using fuzzy matching"""
    user_input_lower = user_input.lower()
    
    # Exact match (case insensitive)
    for team in available_teams:
        if team.lower() == user_input_lower:
            return team
    
    # Check if input is contained in team name or vice versa
    matches = []
    for team in available_teams:
        team_lower = team.lower()
        if user_input_lower in team_lower or team_lower in user_input_lower:
            matches.append(team)
    
    if matches:
        return matches[0]  # Return first match
    
    # Use difflib for fuzzy matching
    close_matches = get_close_matches(user_input, available_teams, n=3, cutoff=0.6)
    if close_matches:
        return close_matches[0]
    
    return None

def get_historical_games(year=2025):
    """Get historical game results for training"""
    with cfbd.ApiClient(configuration) as api_client:
        games_api = cfbd.GamesApi(api_client)
        
        games_data = []
        try:
            # Get games for the year
            games = games_api.get_games(year=year, season_type='regular')
            
            for game in games:
                if game.completed and hasattr(game, 'home_team') and hasattr(game, 'away_team'):
                    # Determine winner
                    if hasattr(game, 'home_points') and hasattr(game, 'away_points'):
                        home_points = game.home_points if game.home_points else 0
                        away_points = game.away_points if game.away_points else 0
                        
                        if home_points > away_points:
                            winner = game.home_team
                            loser = game.away_team
                        elif away_points > home_points:
                            winner = game.away_team
                            loser = game.home_team
                        else:
                            continue  # Skip ties
                        
                        games_data.append({
                            'team1': game.home_team,
                            'team2': game.away_team,
                            'team1_points': home_points,
                            'team2_points': away_points,
                            'winner': winner,
                            'loser': loser
                        })
        except ApiException as e:
            print(f"Error fetching games: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame(games_data)

def prepare_training_data(year=2025):
    """Prepare training data by combining team stats with game results"""
    print(f"Preparing training data for {year} season...")
    
    # Get team stats
    records_df = get_team_records(year)
    stats_df = get_team_stats(year)
    advanced_df = get_advanced_stats(year)
    ratings_df = get_team_ratings(year)
    
    if records_df.empty or stats_df.empty:
        print("Unable to fetch team statistics.")
        return None, None, None
    
    # Merge all stats
    team_stats = pd.merge(records_df, stats_df, on='team', how='inner')
    if not advanced_df.empty:
        team_stats = pd.merge(team_stats, advanced_df, on='team', how='left')
    if not ratings_df.empty:
        team_stats = pd.merge(team_stats, ratings_df, on='team', how='left')
    
    # Calculate efficiency metrics
    if 'games' in team_stats.columns and team_stats['games'].sum() > 0:
        games_col = team_stats['games'].replace(0, 1)
        
        # Check if stats are totals or per-game by sampling
        sample_team = team_stats[team_stats['games'] > 0].iloc[0] if len(team_stats[team_stats['games'] > 0]) > 0 else None
        
        if sample_team is not None and sample_team['games'] > 0:
            points_val = sample_team.get('points_per_game', 0)
            yards_val = sample_team.get('yards_per_game', 0)
            
            # If points > 100 or yards > 5000, they're likely totals
            convert_to_per_game = (points_val > 100) or (yards_val > 5000)
            
            if convert_to_per_game:
                stats_to_convert = [
                    'points_per_game', 'yards_per_game', 'passing_yards_per_game', 'rushing_yards_per_game',
                    'points_allowed_per_game', 'yards_allowed_per_game', 'passing_yards_allowed', 'rushing_yards_allowed',
                    'turnovers', 'fumbles_lost', 'interceptions_thrown', 
                    'takeaways', 'sacks', 'tackles_for_loss', 'interceptions',
                    'fumbles_recovered', 'passes_defended', 'qb_hurries', 'tackles',
                    'pass_attempts', 'pass_completions', 'rush_attempts', 'first_downs',
                    'third_down_conversions', 'third_down_attempts',
                    'fourth_down_conversions', 'fourth_down_attempts',
                    'penalties', 'penalty_yards', 'defensive_penalties', 'defensive_penalty_yards'
                ]
                
                for col in stats_to_convert:
                    if col in team_stats.columns:
                        team_stats[col] = team_stats[col] / games_col
        
        # Calculate efficiency metrics
        if 'pass_attempts' in team_stats.columns and 'pass_completions' in team_stats.columns:
            team_stats['completion_pct'] = (team_stats['pass_completions'] / team_stats['pass_attempts'].replace(0, 1)) * 100
        
        if 'third_down_attempts' in team_stats.columns and 'third_down_conversions' in team_stats.columns:
            team_stats['third_down_pct'] = (team_stats['third_down_conversions'] / team_stats['third_down_attempts'].replace(0, 1)) * 100
        
        if 'takeaways' in team_stats.columns and 'turnovers' in team_stats.columns:
            team_stats['turnover_margin'] = team_stats['takeaways'] - team_stats['turnovers']
        
        if 'yards_per_game' in team_stats.columns and 'points_per_game' in team_stats.columns:
            team_stats['points_per_yard'] = team_stats['points_per_game'] / team_stats['yards_per_game'].replace(0, 1)
        
        if 'yards_allowed_per_game' in team_stats.columns and 'points_allowed_per_game' in team_stats.columns:
            team_stats['yards_allowed_per_point'] = team_stats['yards_allowed_per_game'] / team_stats['points_allowed_per_game'].replace(0, 1)
    
    # Get historical games
    games_df = get_historical_games(year)
    
    if games_df.empty:
        print("No game data available for training.")
        return None, None, None
    
    # Create training examples: for each game, create features comparing the two teams
    training_data = []
    
    for _, game in games_df.iterrows():
        team1_name = game['team1']
        team2_name = game['team2']
        winner = game['winner']
        
        team1_stats = team_stats[team_stats['team'] == team1_name]
        team2_stats = team_stats[team_stats['team'] == team2_name]
        
        if team1_stats.empty or team2_stats.empty:
            continue
        
        team1_stats = team1_stats.iloc[0]
        team2_stats = team2_stats.iloc[0]
        
        # Select important features for comparison (including strength of schedule)
        feature_cols = [
            'points_per_game', 'points_allowed_per_game',
            'yards_per_game', 'yards_allowed_per_game',
            'turnover_margin', 'sacks', 'tackles_for_loss',
            'third_down_pct', 'completion_pct', 'points_per_yard',
            'yards_allowed_per_point', 'win_pct'
        ]
        
        # Add strength of schedule and ratings if available
        if 'strength_of_schedule' in team_stats.columns:
            feature_cols.append('strength_of_schedule')
        if 'sos' in team_stats.columns:  # SRS strength of schedule
            feature_cols.append('sos')
        if 'sp_rating' in team_stats.columns:
            feature_cols.append('sp_rating')
        if 'fpi_rating' in team_stats.columns:
            feature_cols.append('fpi_rating')
        if 'srs_rating' in team_stats.columns:
            feature_cols.append('srs_rating')
        
        # Create difference features (team1 - team2)
        features = {}
        for col in feature_cols:
            if col in team1_stats.index and col in team2_stats.index:
                val1 = team1_stats[col] if pd.notna(team1_stats[col]) else 0
                val2 = team2_stats[col] if pd.notna(team2_stats[col]) else 0
                features[f'{col}_diff'] = val1 - val2
        
        # Label: 1 if team1 wins, 0 if team2 wins
        label = 1 if winner == team1_name else 0
        
        features['label'] = label
        training_data.append(features)
    
    if not training_data:
        print("No training data could be created.")
        return None, None, None
    
    train_df = pd.DataFrame(training_data)
    
    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col != 'label']
    X = train_df[feature_cols].fillna(0)
    y = train_df['label']
    
    return X, y, feature_cols

def train_decision_tree(X, y, max_depth=5):
    """Train a Random Forest classifier (better probability estimates than single tree)"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Random Forest instead of single tree for better probability estimates
    # Random Forest averages over many trees, giving more calibrated probabilities
    clf = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,  # Prevent overfitting
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    print(f"\nRandom Forest Training Accuracy: {train_acc:.3f}")
    print(f"Random Forest Test Accuracy: {test_acc:.3f}")
    
    return clf, list(X.columns)

def predict_game(team1_name, team2_name, clf, feature_names, year=2025):
    """Predict the winner between two teams"""
    print(f"\n{'='*70}")
    print(f"GAME PREDICTION: {team1_name} vs {team2_name}")
    print(f"{'='*70}")
    
    # Get team stats
    records_df = get_team_records(year)
    stats_df = get_team_stats(year)
    advanced_df = get_advanced_stats(year)
    
    if records_df.empty or stats_df.empty:
        print("Unable to fetch team statistics.")
        return None
    
    # Get team ratings (includes strength of schedule)
    print("Fetching team ratings and strength of schedule...")
    ratings_df = get_team_ratings(year)
    
    # Merge stats
    team_stats = pd.merge(records_df, stats_df, on='team', how='inner')
    if not advanced_df.empty:
        team_stats = pd.merge(team_stats, advanced_df, on='team', how='left')
    if not ratings_df.empty:
        team_stats = pd.merge(team_stats, ratings_df, on='team', how='left')
        print(f"Added ratings data for {len(ratings_df)} teams")
    
    # Calculate efficiency metrics
    # Check if API returns totals or per-game by examining values
    if 'games' in team_stats.columns and team_stats['games'].sum() > 0:
        games_col = team_stats['games'].replace(0, 1)
        
        # Check if stats are already per-game or totals by sampling
        # If median points > 100, likely totals; if < 50, likely per-game
        sample_team = team_stats[team_stats['games'] > 0].iloc[0] if len(team_stats[team_stats['games'] > 0]) > 0 else None
        
        if sample_team is not None and sample_team['games'] > 0:
            # Check if points_per_game looks like totals (very high) or per-game (reasonable)
            points_val = sample_team.get('points_per_game', 0)
            yards_val = sample_team.get('yards_per_game', 0)
            
            # If points > 100 or yards > 5000, they're likely totals
            convert_to_per_game = (points_val > 100) or (yards_val > 5000)
            
            if convert_to_per_game:
                # Convert totals to per-game
                stats_to_convert = [
                    'points_per_game', 'yards_per_game', 'passing_yards_per_game', 'rushing_yards_per_game',
                    'points_allowed_per_game', 'yards_allowed_per_game', 'passing_yards_allowed', 'rushing_yards_allowed',
                    'turnovers', 'fumbles_lost', 'interceptions_thrown', 
                    'takeaways', 'sacks', 'tackles_for_loss', 'interceptions',
                    'fumbles_recovered', 'passes_defended', 'qb_hurries', 'tackles',
                    'pass_attempts', 'pass_completions', 'rush_attempts', 'first_downs',
                    'third_down_conversions', 'third_down_attempts',
                    'fourth_down_conversions', 'fourth_down_attempts',
                    'penalties', 'penalty_yards', 'defensive_penalties', 'defensive_penalty_yards'
                ]
                
                for col in stats_to_convert:
                    if col in team_stats.columns:
                        team_stats[col] = team_stats[col] / games_col
        
        # Calculate efficiency metrics
        if 'pass_attempts' in team_stats.columns and 'pass_completions' in team_stats.columns:
            team_stats['completion_pct'] = (team_stats['pass_completions'] / team_stats['pass_attempts'].replace(0, 1)) * 100
        
        if 'third_down_attempts' in team_stats.columns and 'third_down_conversions' in team_stats.columns:
            team_stats['third_down_pct'] = (team_stats['third_down_conversions'] / team_stats['third_down_attempts'].replace(0, 1)) * 100
        
        if 'takeaways' in team_stats.columns and 'turnovers' in team_stats.columns:
            team_stats['turnover_margin'] = team_stats['takeaways'] - team_stats['turnovers']
        
        if 'yards_per_game' in team_stats.columns and 'points_per_game' in team_stats.columns:
            team_stats['points_per_yard'] = team_stats['points_per_game'] / team_stats['yards_per_game'].replace(0, 1)
        
        if 'yards_allowed_per_game' in team_stats.columns and 'points_allowed_per_game' in team_stats.columns:
            team_stats['yards_allowed_per_point'] = team_stats['yards_allowed_per_game'] / team_stats['points_allowed_per_game'].replace(0, 1)
    
    # Get all available teams for matching
    available_teams = sorted(team_stats['team'].tolist())
    
    # Try to find matching team names
    team1_found = find_team_name(team1_name, available_teams)
    team2_found = find_team_name(team2_name, available_teams)
    
    if not team1_found:
        print(f"\nError: Could not find team '{team1_name}'")
        print(f"\nDid you mean one of these?")
        # Show suggestions
        suggestions = get_close_matches(team1_name, available_teams, n=5, cutoff=0.4)
        for i, sug in enumerate(suggestions, 1):
            print(f"  {i}. {sug}")
        print(f"\nSearching for teams containing '{team1_name}'...")
        containing = [t for t in available_teams if team1_name.lower() in t.lower()]
        if containing:
            for t in containing[:5]:
                print(f"  - {t}")
        return None
    
    if not team2_found:
        print(f"\nError: Could not find team '{team2_name}'")
        print(f"\nDid you mean one of these?")
        suggestions = get_close_matches(team2_name, available_teams, n=5, cutoff=0.4)
        for i, sug in enumerate(suggestions, 1):
            print(f"  {i}. {sug}")
        print(f"\nSearching for teams containing '{team2_name}'...")
        containing = [t for t in available_teams if team2_name.lower() in t.lower()]
        if containing:
            for t in containing[:5]:
                print(f"  - {t}")
        return None
    
    # Use the found team names
    if team1_found != team1_name:
        print(f"Using '{team1_found}' for '{team1_name}'")
    if team2_found != team2_name:
        print(f"Using '{team2_found}' for '{team2_name}'")
    
    team1_stats = team_stats[team_stats['team'] == team1_found]
    team2_stats = team_stats[team_stats['team'] == team2_found]
    
    # Update names for display
    team1_name = team1_found
    team2_name = team2_found
    
    team1_stats = team1_stats.iloc[0]
    team2_stats = team2_stats.iloc[0]
    
    # Create feature vector (differences)
    feature_dict = {}
    for feat in feature_names:
        # Extract the base stat name (remove _diff)
        base_stat = feat.replace('_diff', '')
        if base_stat in team1_stats.index and base_stat in team2_stats.index:
            val1 = team1_stats[base_stat] if pd.notna(team1_stats[base_stat]) else 0
            val2 = team2_stats[base_stat] if pd.notna(team2_stats[base_stat]) else 0
            feature_dict[feat] = val1 - val2
    
    # Create DataFrame with same column order as training
    feature_df = pd.DataFrame([feature_dict])[feature_names].fillna(0)
    
    # Make prediction
    prediction = clf.predict(feature_df)[0]
    probabilities = clf.predict_proba(feature_df)[0]
    
    # Apply probability smoothing to prevent overconfidence
    # Add a small amount of uncertainty (Laplace smoothing)
    smoothing_factor = 0.05  # 5% uncertainty floor
    probabilities_smooth = np.array(probabilities)
    probabilities_smooth = probabilities_smooth * (1 - 2 * smoothing_factor) + smoothing_factor
    
    # Renormalize
    probabilities_smooth = probabilities_smooth / probabilities_smooth.sum()
    
    # Use smoothed probabilities for display, but original for prediction
    winner = team1_name if prediction == 1 else team2_name
    winner_prob_smooth = probabilities_smooth[1] if prediction == 1 else probabilities_smooth[0]
    
    print(f"\nPREDICTED WINNER: {winner}")
    print(f"Confidence: {winner_prob_smooth:.1%}")
    print(f"\n{team1_name} win probability: {probabilities_smooth[1]:.1%}")
    print(f"{team2_name} win probability: {probabilities_smooth[0]:.1%}")
    
    # Show raw probabilities in parentheses for transparency
    if abs(probabilities[0] - probabilities_smooth[0]) > 0.01:
        print(f"\n(Raw model probabilities: {team1_name} {probabilities[1]:.1%}, {team2_name} {probabilities[0]:.1%})")
    
    # Show key stat comparisons
    print(f"\n{'='*70}")
    print("KEY STATISTIC COMPARISONS:")
    print(f"{'='*70}")
    
    key_stats = {
        'points_per_game': 'Points Per Game',
        'points_allowed_per_game': 'Points Allowed Per Game',
        'yards_per_game': 'Yards Per Game',
        'yards_allowed_per_game': 'Yards Allowed Per Game',
        'turnover_margin': 'Turnover Margin',
        'win_pct': 'Win Percentage',
        'third_down_pct': 'Third Down %',
        'completion_pct': 'Completion %',
        'sacks': 'Sacks Per Game',
        'tackles_for_loss': 'TFL Per Game',
        'strength_of_schedule': 'Strength of Schedule (FPI)',
        'sos': 'Strength of Schedule (SRS)',
        'sp_rating': 'SP+ Rating',
        'fpi_rating': 'FPI Rating',
        'srs_rating': 'SRS Rating'
    }
    
    print(f"\n{'Statistic':<30} {team1_name:<25} {team2_name:<25} {'Advantage':<15}")
    print("-" * 95)
    
    for stat, label in key_stats.items():
        if stat in team1_stats.index and stat in team2_stats.index:
            val1 = team1_stats[stat] if pd.notna(team1_stats[stat]) and not np.isinf(team1_stats[stat]) else 0
            val2 = team2_stats[stat] if pd.notna(team2_stats[stat]) and not np.isinf(team2_stats[stat]) else 0
            
            
            # Determine advantage
            if stat in ['points_allowed_per_game', 'yards_allowed_per_game']:
                # Lower is better
                if val1 < val2:
                    adv = f"→ {team1_name}"
                elif val2 < val1:
                    adv = f"→ {team2_name}"
                else:
                    adv = "Even"
            else:
                # Higher is better
                if val1 > val2:
                    adv = f"→ {team1_name}"
                elif val2 > val1:
                    adv = f"→ {team2_name}"
                else:
                    adv = "Even"
            
            # Format values
            if 'pct' in stat:
                val1_str = f"{val1:.1f}%"
                val2_str = f"{val2:.1f}%"
            elif stat == 'win_pct':
                val1_str = f"{val1:.3f}"
                val2_str = f"{val2:.3f}"
            else:
                val1_str = f"{val1:.1f}"
                val2_str = f"{val2:.1f}"
            
            print(f"{label:<30} {val1_str:<25} {val2_str:<25} {adv:<15}")
    
    # Show feature importance from decision tree
    print(f"\n{'='*70}")
    print("MOST IMPORTANT FACTORS IN THIS PREDICTION:")
    print(f"{'='*70}")
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        base_stat = row['feature'].replace('_diff', '').replace('_', ' ').title()
        print(f"  {base_stat:<40} {row['importance']:.3f}")
    
    return winner, winner_prob_smooth

def main():
    """Main function to train model and make predictions"""
    year = 2025
    
    print("Training Decision Tree Model...")
    print("="*70)
    
    # Prepare training data
    result = prepare_training_data(year)
    
    if result[0] is None or result[1] is None:
        print("Failed to prepare training data. Check your API token.")
        print("\nTo set your API token, run:")
        print('  $env:BEARER_TOKEN = "your-api-key-here"')
        return
    
    X, y, feature_names = result
    
    print(f"\nTraining on {len(X)} games...")
    
    # Train model
    clf, trained_feature_names = train_decision_tree(X, y, max_depth=5)
    feature_names = trained_feature_names
    
    # Get available teams for reference
    available_teams = get_all_teams(year)
    
    # Interactive prediction
    print("\n" + "="*70)
    print("GAME PREDICTOR READY!")
    print("="*70)
    print("\nEnter two team names to predict the winner.")
    print("Type 'list' to see all available teams.")
    print("Type 'search <name>' to search for a team.")
    print("Type 'quit' to exit.\n")
    
    while True:
        team1 = input("Enter Team 1 name: ").strip()
        if team1.lower() == 'quit':
            break
        elif team1.lower() == 'list':
            print(f"\nAvailable teams ({len(available_teams)}):")
            for i, team in enumerate(available_teams, 1):
                print(f"  {team}", end="")
                if i % 5 == 0:
                    print()
                else:
                    print(" | ", end="")
            print("\n")
            continue
        elif team1.lower().startswith('search '):
            search_term = team1[7:].strip()
            matches = [t for t in available_teams if search_term.lower() in t.lower()]
            if matches:
                print(f"\nTeams matching '{search_term}':")
                for match in matches[:10]:
                    print(f"  - {match}")
            else:
                print(f"\nNo teams found matching '{search_term}'")
            print()
            continue
        
        team2 = input("Enter Team 2 name: ").strip()
        if team2.lower() == 'quit':
            break
        elif team2.lower() == 'list':
            print(f"\nAvailable teams ({len(available_teams)}):")
            for i, team in enumerate(available_teams, 1):
                print(f"  {team}", end="")
                if i % 5 == 0:
                    print()
                else:
                    print(" | ", end="")
            print("\n")
            continue
        elif team2.lower().startswith('search '):
            search_term = team2[7:].strip()
            matches = [t for t in available_teams if search_term.lower() in t.lower()]
            if matches:
                print(f"\nTeams matching '{search_term}':")
                for match in matches[:10]:
                    print(f"  - {match}")
            else:
                print(f"\nNo teams found matching '{search_term}'")
            print()
            continue
        
        try:
            predict_game(team1, team2, clf, feature_names, year)
            print("\n" + "-"*70 + "\n")
        except Exception as e:
            print(f"Error making prediction: {e}\n")

if __name__ == "__main__":
    main()

