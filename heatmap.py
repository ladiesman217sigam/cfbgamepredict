import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cfbd
from cfbd.rest import ApiException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure API
configuration = cfbd.Configuration(
    host="https://api.collegefootballdata.com"
)

# Get API token from environment variable (.env file)
bearer_token = os.environ.get("BEARER_TOKEN", "")
if bearer_token:
    configuration.access_token = bearer_token
else:
    print("Warning: No BEARER_TOKEN found in environment variables or .env file.")
    print("Please create a .env file with your API key:")
    print("  BEARER_TOKEN=your-api-key-here")
    print("Get a free API key at: https://collegefootballdata.com\n")

def get_team_records(year=2025):
    """Get team records and win percentages"""
    with cfbd.ApiClient(configuration) as api_client:
        games_api = cfbd.GamesApi(api_client)
        
        records_data = []
        try:
            # Get records for the year
            records = games_api.get_records(year=year)
            
            for record in records:
                team = record.team
                total_games = record.total.games
                total_wins = record.total.wins
                win_pct = total_wins / total_games if total_games > 0 else 0
                
                records_data.append({
                    'team': team,
                    'wins': total_wins,
                    'losses': record.total.losses,
                    'games': total_games,
                    'win_pct': win_pct
                })
        except ApiException as e:
            if e.status == 401:
                print(f"Authentication required. Please set BEARER_TOKEN environment variable.")
            else:
                print(f"Error fetching records: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error fetching records: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame(records_data)

def get_team_stats(year=2025):
    """Get comprehensive team statistics including offense, defense, and advanced metrics"""
    with cfbd.ApiClient(configuration) as api_client:
        stats_api = cfbd.StatsApi(api_client)
        
        stats_data = []
        try:
            # Get team stats
            team_stats = stats_api.get_team_stats(year=year)
            
            for stat in team_stats:
                stat_dict = {
                    'team': stat.team if hasattr(stat, 'team') else '',
                    'games': stat.games if hasattr(stat, 'games') else 0,
                }
                
                # Offensive stats
                if hasattr(stat, 'offense') and stat.offense:
                    off = stat.offense
                    stat_dict.update({
                        # Basic offensive stats
                        'points_per_game': getattr(off, 'points', 0),
                        'yards_per_game': getattr(off, 'yards', 0),
                        'passing_yards_per_game': getattr(off, 'passingYards', 0),
                        'rushing_yards_per_game': getattr(off, 'rushingYards', 0),
                        'turnovers': getattr(off, 'turnovers', 0),
                        'fumbles_lost': getattr(off, 'fumbles', 0),
                        'interceptions_thrown': getattr(off, 'interceptions', 0),
                        # Advanced offensive stats
                        'pass_attempts': getattr(off, 'passAttempts', 0),
                        'pass_completions': getattr(off, 'completions', 0),
                        'rush_attempts': getattr(off, 'rushAttempts', 0),
                        'first_downs': getattr(off, 'firstDowns', 0),
                        'third_down_conversions': getattr(off, 'thirdDownConversions', 0),
                        'third_down_attempts': getattr(off, 'thirdDownAttempts', 0),
                        'fourth_down_conversions': getattr(off, 'fourthDownConversions', 0),
                        'fourth_down_attempts': getattr(off, 'fourthDownAttempts', 0),
                        'penalties': getattr(off, 'penalties', 0),
                        'penalty_yards': getattr(off, 'penaltyYards', 0),
                        'time_of_possession_seconds': getattr(off, 'possessionTime', 0),
                    })
                else:
                    # Default values
                    defaults = {
                        'points_per_game': 0, 'yards_per_game': 0,
                        'passing_yards_per_game': 0, 'rushing_yards_per_game': 0,
                        'turnovers': 0, 'fumbles_lost': 0, 'interceptions_thrown': 0,
                        'pass_attempts': 0, 'pass_completions': 0, 'rush_attempts': 0,
                        'first_downs': 0, 'third_down_conversions': 0, 'third_down_attempts': 0,
                        'fourth_down_conversions': 0, 'fourth_down_attempts': 0,
                        'penalties': 0, 'penalty_yards': 0, 'time_of_possession_seconds': 0,
                    }
                    stat_dict.update(defaults)
                
                # Defensive stats
                if hasattr(stat, 'defense') and stat.defense:
                    def_stat = stat.defense
                    stat_dict.update({
                        # Basic defensive stats
                        'points_allowed_per_game': getattr(def_stat, 'points', 0),
                        'yards_allowed_per_game': getattr(def_stat, 'yards', 0),
                        'passing_yards_allowed': getattr(def_stat, 'passingYards', 0),
                        'rushing_yards_allowed': getattr(def_stat, 'rushingYards', 0),
                        'takeaways': getattr(def_stat, 'turnovers', 0),
                        'sacks': getattr(def_stat, 'sacks', 0),
                        'tackles_for_loss': getattr(def_stat, 'tacklesForLoss', 0),
                        # Advanced defensive stats
                        'interceptions': getattr(def_stat, 'interceptions', 0),
                        'fumbles_recovered': getattr(def_stat, 'fumbles', 0),
                        'passes_defended': getattr(def_stat, 'passesDefended', 0),
                        'qb_hurries': getattr(def_stat, 'qbHurries', 0),
                        'tackles': getattr(def_stat, 'tackles', 0),
                        'defensive_penalties': getattr(def_stat, 'penalties', 0),
                        'defensive_penalty_yards': getattr(def_stat, 'penaltyYards', 0),
                    })
                else:
                    defaults = {
                        'points_allowed_per_game': 0, 'yards_allowed_per_game': 0,
                        'passing_yards_allowed': 0, 'rushing_yards_allowed': 0,
                        'takeaways': 0, 'sacks': 0, 'tackles_for_loss': 0,
                        'interceptions': 0, 'fumbles_recovered': 0, 'passes_defended': 0,
                        'qb_hurries': 0, 'tackles': 0, 'defensive_penalties': 0,
                        'defensive_penalty_yards': 0,
                    }
                    stat_dict.update(defaults)
                
                stats_data.append(stat_dict)
        except ApiException as e:
            if e.status == 401:
                print(f"Authentication required. Please set BEARER_TOKEN environment variable.")
            else:
                print(f"Error fetching stats: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error fetching stats: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame(stats_data)

def get_advanced_stats(year=2025):
    """Get advanced team statistics and efficiency metrics"""
    with cfbd.ApiClient(configuration) as api_client:
        stats_api = cfbd.StatsApi(api_client)
        
        advanced_data = []
        try:
            # Get advanced season stats
            advanced_stats = stats_api.get_advanced_season_stats(year=year)
            
            for stat in advanced_stats:
                adv_dict = {
                    'team': stat.team if hasattr(stat, 'team') else '',
                }
                
                # Offensive efficiency
                if hasattr(stat, 'offense') and stat.offense:
                    off_adv = stat.offense
                    adv_dict.update({
                        'off_plays': getattr(off_adv, 'plays', 0),
                        'off_line_yards': getattr(off_adv, 'lineYards', 0),
                        'off_line_yards_per_carry': getattr(off_adv, 'lineYardsPerCarry', 0),
                        'off_second_level_yards': getattr(off_adv, 'secondLevelYards', 0),
                        'off_open_field_yards': getattr(off_adv, 'openFieldYards', 0),
                        'off_standard_downs_rate': getattr(off_adv, 'standardDownsRate', 0),
                        'off_passing_downs_rate': getattr(off_adv, 'passingDownsRate', 0),
                        'off_explosiveness': getattr(off_adv, 'explosiveness', 0),
                        'off_power_success': getattr(off_adv, 'powerSuccess', 0),
                        'off_stuff_rate': getattr(off_adv, 'stuffRate', 0),
                        'off_sack_rate': getattr(off_adv, 'sackRate', 0),
                        'off_passing_plays_rate': getattr(off_adv, 'passingPlaysRate', 0),
                    })
                
                # Defensive efficiency
                if hasattr(stat, 'defense') and stat.defense:
                    def_adv = stat.defense
                    adv_dict.update({
                        'def_plays': getattr(def_adv, 'plays', 0),
                        'def_line_yards': getattr(def_adv, 'lineYards', 0),
                        'def_line_yards_per_carry': getattr(def_adv, 'lineYardsPerCarry', 0),
                        'def_second_level_yards': getattr(def_adv, 'secondLevelYards', 0),
                        'def_open_field_yards': getattr(def_adv, 'openFieldYards', 0),
                        'def_standard_downs_rate': getattr(def_adv, 'standardDownsRate', 0),
                        'def_passing_downs_rate': getattr(def_adv, 'passingDownsRate', 0),
                        'def_explosiveness': getattr(def_adv, 'explosiveness', 0),
                        'def_power_success': getattr(def_adv, 'powerSuccess', 0),
                        'def_stuff_rate': getattr(def_adv, 'stuffRate', 0),
                        'def_sack_rate': getattr(def_adv, 'sackRate', 0),
                        'def_passing_plays_rate': getattr(def_adv, 'passingPlaysRate', 0),
                    })
                
                advanced_data.append(adv_dict)
        except ApiException as e:
            # Advanced stats might not be available, return empty
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()
    
    return pd.DataFrame(advanced_data)

def create_win_correlation_heatmap(year=2025):
    """Create a heatmap showing correlation between stats and win percentage"""
    print(f"Fetching data for {year} season...")
    
    # Get team records
    records_df = get_team_records(year)
    if records_df.empty:
        print("No records data available. Trying without API token...")
        return
    
    # Get team stats
    stats_df = get_team_stats(year)
    if stats_df.empty:
        print("No stats data available. Trying without API token...")
        return
    
    # Get advanced stats
    print("Fetching advanced statistics...")
    advanced_df = get_advanced_stats(year)
    
    # Merge records and stats
    df = pd.merge(records_df, stats_df, on='team', how='inner')
    
    # Merge advanced stats if available
    if not advanced_df.empty:
        df = pd.merge(df, advanced_df, on='team', how='left')
        print(f"Added {len(advanced_df.columns)-1} advanced statistics")
    
    if df.empty:
        print("No data to analyze. Check your API token or try a different year.")
        return
    
    # Calculate per-game averages and efficiency metrics
    if 'games' in df.columns and df['games'].sum() > 0:
        games_col = df['games'].replace(0, 1)  # Avoid division by zero
        
        # Stats that are totals - convert to per-game
        total_stats = ['turnovers', 'fumbles_lost', 'interceptions_thrown', 
                      'takeaways', 'sacks', 'tackles_for_loss', 'interceptions',
                      'fumbles_recovered', 'passes_defended', 'qb_hurries', 'tackles',
                      'pass_attempts', 'pass_completions', 'rush_attempts', 'first_downs',
                      'third_down_conversions', 'third_down_attempts',
                      'fourth_down_conversions', 'fourth_down_attempts',
                      'penalties', 'penalty_yards', 'defensive_penalties', 'defensive_penalty_yards']
        
        for col in total_stats:
            if col in df.columns:
                df[col] = df[col] / games_col
        
        # Calculate efficiency metrics
        # Completion percentage
        if 'pass_attempts' in df.columns and 'pass_completions' in df.columns:
            df['completion_pct'] = (df['pass_completions'] / df['pass_attempts'].replace(0, 1)) * 100
        
        # Third down conversion rate
        if 'third_down_attempts' in df.columns and 'third_down_conversions' in df.columns:
            df['third_down_pct'] = (df['third_down_conversions'] / df['third_down_attempts'].replace(0, 1)) * 100
        
        # Fourth down conversion rate
        if 'fourth_down_attempts' in df.columns and 'fourth_down_conversions' in df.columns:
            df['fourth_down_pct'] = (df['fourth_down_conversions'] / df['fourth_down_attempts'].replace(0, 1)) * 100
        
        # Yards per pass attempt
        if 'pass_attempts' in df.columns and 'passing_yards_per_game' in df.columns:
            df['yards_per_pass_attempt'] = df['passing_yards_per_game'] / df['pass_attempts'].replace(0, 1)
        
        # Yards per rush attempt
        if 'rush_attempts' in df.columns and 'rushing_yards_per_game' in df.columns:
            df['yards_per_rush_attempt'] = df['rushing_yards_per_game'] / df['rush_attempts'].replace(0, 1)
        
        # Turnover margin (per game)
        if 'takeaways' in df.columns and 'turnovers' in df.columns:
            df['turnover_margin'] = df['takeaways'] - df['turnovers']
        
        # Time of possession (convert seconds to minutes per game)
        if 'time_of_possession_seconds' in df.columns:
            df['time_of_possession_minutes'] = df['time_of_possession_seconds'] / 60
        
        # Penalty yards per penalty
        if 'penalties' in df.columns and 'penalty_yards' in df.columns:
            df['penalty_yards_per_penalty'] = df['penalty_yards'] / df['penalties'].replace(0, 1)
        
        # Points per yard (scoring efficiency)
        if 'yards_per_game' in df.columns and 'points_per_game' in df.columns:
            df['points_per_yard'] = df['points_per_game'] / df['yards_per_game'].replace(0, 1)
        
        # Yards allowed per point (defensive efficiency)
        if 'yards_allowed_per_game' in df.columns and 'points_allowed_per_game' in df.columns:
            df['yards_allowed_per_point'] = df['yards_allowed_per_game'] / df['points_allowed_per_game'].replace(0, 1)
    
    # Select numeric columns for correlation
    stat_columns = [col for col in df.columns if col not in ['team', 'wins', 'losses', 'games', 'win_pct']]
    
    # Calculate correlations with win percentage
    correlations = {}
    for col in stat_columns:
        if df[col].dtype in [np.float64, np.int64]:
            # Remove any invalid values
            valid_mask = ~(np.isnan(df[col]) | np.isinf(df[col]))
            if valid_mask.sum() > 5:  # Need at least 5 valid data points
                corr = df.loc[valid_mask, 'win_pct'].corr(df.loc[valid_mask, col])
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations[col] = corr
    
    # Debug: Print some sample data to verify
    print(f"\nData summary:")
    print(f"Teams: {len(df)}")
    print(f"Win % range: {df['win_pct'].min():.3f} - {df['win_pct'].max():.3f}")
    if 'points_per_game' in df.columns:
        print(f"Points per game range: {df['points_per_game'].min():.1f} - {df['points_per_game'].max():.1f}")
    if 'points_allowed_per_game' in df.columns:
        print(f"Points allowed range: {df['points_allowed_per_game'].min():.1f} - {df['points_allowed_per_game'].max():.1f}")
    
    if not correlations:
        print("No valid correlations found. Check your data.")
        return
    
    # Create correlation matrix
    corr_data = pd.DataFrame(list(correlations.items()), columns=['Stat', 'Correlation'])
    corr_data = corr_data.sort_values('Correlation', key=abs, ascending=False)
    
    # Verify we have both positive and negative correlations
    print(f"\nCorrelation range: {corr_data['Correlation'].min():.3f} to {corr_data['Correlation'].max():.3f}")
    print(f"Positive correlations: {(corr_data['Correlation'] > 0).sum()}")
    print(f"Negative correlations: {(corr_data['Correlation'] < 0).sum()}")
    
    # Create a matrix for heatmap (stats vs correlation)
    heatmap_data = corr_data.set_index('Stat').T
    
    # Create the heatmap with better sizing for many variables
    num_stats = len(corr_data)
    fig_width = max(20, num_stats * 0.5)  # Dynamic width based on number of stats
    plt.figure(figsize=(fig_width, 12))
    
    # Format stat names for better readability
    def format_stat_name(stat):
        # Replace underscores and format nicely
        formatted = stat.replace('_', ' ').title()
        # Add line breaks for long names
        if len(formatted) > 20:
            words = formatted.split()
            mid = len(words) // 2
            formatted = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        return formatted
    
    formatted_stats = [format_stat_name(stat) for stat in corr_data['Stat'].values]
    
    # Create a better matrix format
    heatmap_matrix = corr_data['Correlation'].values.reshape(1, -1)
    
    # Use a diverging colormap that clearly shows positive and negative
    sns.heatmap(heatmap_matrix, 
                xticklabels=formatted_stats,
                yticklabels=['Correlation'],
                annot=True, 
                fmt='.3f', 
                cmap='RdBu_r',  # Red-Blue reversed: Blue=positive, Red=negative
                center=0,
                vmin=-1, 
                vmax=1,
                cbar_kws={'label': 'Correlation with Win %', 'shrink': 0.8},
                linewidths=0.5,
                annot_kws={'fontsize': 8, 'weight': 'bold'},
                square=False)
    
    plt.title(f'Correlation Between Team Statistics and Win Percentage ({year} Season)\n' +
              'Blue = Positive correlation (Higher stat → More wins) | Red = Negative correlation (Lower stat → More wins)', 
              fontsize=13, fontweight='bold', pad=20)
    plt.xlabel('Team Statistics', fontsize=12, fontweight='bold')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('win_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved as 'win_correlation_heatmap.png'")
    
    # Categorize stats for better organization
    stat_categories = {
        'Scoring': ['points_per_game', 'points_allowed_per_game', 'points_per_yard'],
        'Yardage': ['yards_per_game', 'yards_allowed_per_game', 'passing_yards_per_game', 
                   'rushing_yards_per_game', 'yards_per_pass_attempt', 'yards_per_rush_attempt'],
        'Turnovers': ['turnovers', 'takeaways', 'turnover_margin', 'interceptions_thrown', 
                     'fumbles_lost', 'interceptions', 'fumbles_recovered'],
        'Efficiency': ['completion_pct', 'third_down_pct', 'fourth_down_pct', 
                      'yards_allowed_per_point', 'off_explosiveness', 'def_explosiveness'],
        'Defense': ['sacks', 'tackles_for_loss', 'passes_defended', 'qb_hurries', 
                   'tackles', 'def_sack_rate', 'def_stuff_rate'],
        'Offense': ['first_downs', 'pass_attempts', 'rush_attempts', 
                   'off_power_success', 'off_sack_rate'],
        'Special Teams/Other': ['penalties', 'penalty_yards', 'time_of_possession_minutes'],
        'Advanced': ['off_line_yards_per_carry', 'def_line_yards_per_carry', 
                    'off_standard_downs_rate', 'def_standard_downs_rate']
    }
    
    # Print top correlations
    print("\n" + "="*70)
    print("TOP STATISTICS CORRELATED WITH WINNING:")
    print("="*70)
    print("\nPositive Correlations (Higher = Better for Winning):")
    positive = corr_data[corr_data['Correlation'] > 0].sort_values('Correlation', ascending=False)
    for _, row in positive.head(15).iterrows():
        # Find category
        category = 'Other'
        for cat, stats in stat_categories.items():
            if row['Stat'] in stats:
                category = cat
                break
        print(f"  {row['Stat']:.<45} {row['Correlation']:>6.3f} [{category}]")
    
    print("\nNegative Correlations (Lower = Better for Winning):")
    negative = corr_data[corr_data['Correlation'] < 0].sort_values('Correlation', ascending=True)
    for _, row in negative.head(15).iterrows():
        # Find category
        category = 'Other'
        for cat, stats in stat_categories.items():
            if row['Stat'] in stats:
                category = cat
                break
        print(f"  {row['Stat']:.<45} {row['Correlation']:>6.3f} [{category}]")
    
    print(f"\nTotal variables analyzed: {len(corr_data)}")
    
    plt.show()

if __name__ == "__main__":
    # Try 2025, then fallback to recent years if needed
    for year in [2025, 2024, 2023]:
        try:
            create_win_correlation_heatmap(year)
            break
        except Exception as e:
            print(f"Error with year {year}: {e}")
            continue
