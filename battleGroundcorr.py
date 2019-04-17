import pandas as pd
from pandas import DataFrame as df



# 1. 데이터불러오기 ( 데이터 요약및 살펴보기 및. 도식화 해보기)
# 2. train 파일  test 파일 나누기 
# 3. 정규화 및 이상치 결측치 제거하기
# 4-1.  어떤 방식으로 어떤 알고리즘, 분류방식으로 할지 결정하기
# 4-2. 훈련시키기 - 모델생성
# 5  예측밎 결과 치 내기

pd.set_option('display.expand_frame_repr', False) # 출력값 요약 말고 다보이게 하는 코드 

# 1. 트레인 테스트 파일 분리하기
train = pd.read_csv('train_V2.csv') # 파일 읽어 오기 

# train = train[:500]


print('train.info값', train.info())  # 결측치 없음
# print(train.describe(include='all'))

# pd.unique = 중복되는값 제거
print(len(pd.unique(train['Id']))) # 아이디컬럼의 중복값제외한 개수 
print(len(pd.unique(train.Id))) # 아이디컬럼의 중복값제외한 개수
print(len(train.Id)) # 중복 포함 개수 

print(train.groupby("matchId").size()) # matchID 컬럼을 기준으로 출력하고, 매치아이디의 개수를 출력해라.
# wow = train.loc[train.matchId == "292611730ca862"] # matchID컬럼 중 = 의 값만을 가져와서 wow에 담아라 
# print(wow.groupby("matchId").size())
# print('sum',sum(train.groupby('matchId').size())) # 합
# print(pd.unique(train.matchId))

# 퍼셉트론의 3단계  1. dot product 2. add a bias. 3. taking a non linearity line

# 변수 해석
# 
# DBNOs - Number of enemy players knocked. / 적 기절 시킨 숫자 / 중
# assists - Number of enemy players this player damaged that were killed by teammates. / 동료에 의해 죽게 데미지 입힌 상황에 숫자  / 중
# boosts - Number of boost items used. / 부스트 아이템 사용한 숫자  
# damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.  / 총 데미지 입힌 수치
# headshotKills - Number of enemy players killed with headshots. / - 헤드샷 한 숫자
# heals - Number of healing items used./ 힐아이템 사용한 숫자
# Id - Player’s Id ./ 아디
# killPlace - Ranking in match of number of enemy players killed. 가장 많은 킬한 사람 순위  / 1이 가장 많은 킬한 순위
# killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) 
#            - If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
#            - 외부적인 플레이어의 랭킹 / 킬에 기반한 / 킬으로만 랭킹 매김 / 
# killStreaks - Max number of enemy players killed in a short amount of time. 짧은 찰라에 죽인 적의 최대숫자
 
# kills - Number of enemy players killed. / 킬딴 숫자
# longestKill - Longest distance between player and player killed at time of death. / 가장 먼 거리에서 죽인 거리
#             - This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# matchDuration - Duration of match in seconds. / 초 기준으로 경기 시간
# matchId - ID to identify match. There are no matches that are in both the training and testing set. / 각 경기를 구분하기 위한 아이디
# matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version,
#  so use with caution. Value of -1 takes place of “None”.
# revives - Number of times this player revived teammates. 리바이브한 숫자 
# rideDistance - Total distance traveled in vehicles measured in meters. 탈것으로 이동한 거리 
# roadKills - Number of kills while in a vehicle. 로드킬한 숫자
# swimDistance - Total distance traveled by swimming measured in meters. 수영거리
# teamKills - Number of times this player killed a teammate. 팀킬한 숫자
# vehicleDestroys - Number of vehicles destroyed. 파괴한 자동차 숫자 
# walkDistance - Total distance traveled on foot measured in meters. 움직인 총 거리
# weaponsAcquired - Number of weapons picked up. / 무기 획득한 숫자 
# winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) 
#             If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# numGroups - Number of groups we have data for in the match. 우리가 시합에서 데이터를 가지고 있는 그룹의 수.
# maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
            # 최악의 장소 우리가 경기에서 가지고 있는 / 이것은 numGroup과 맞지 않을 수도 있다/ 떄떄로 데이터가 스킵되기에
            #우리가 그 시합에서 데이터를 가지고 있는 최악의 장소. 때로는 데이터가 배치 위로 건너뛰기 때문에 이것은 numGroups와 일치하지 않을 수 있다.

# winPlacePerc - The target of prediction. This is a percentile winning placement,  
#                where 1 corresponds to 1st place, and 0 corresponds to last place in the match. 1에 가까우면 1위 0에 가까우면 꼴찌 
#                 It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# 측의 대상. 이것은 백분위수 승점입니다, 
# 여기서 1은 1위에 해당하며, 0은 시합의 꼴찌에 해당한다. 
# 이것은 numGroups가 아닌 maxPlace로 계산되므로, 매치에서 누락된 청크를 가질 수 있다.

# 예측의 대상. 이는 백분위 수위를 차지하는 배치이며,
# 1은 1 위를 나타내고 0은 마지막 경기를 나타냅니다.
# 그것은 numGroups가 아니라 maxPlace에서 계산되므로 매치에서 덩어리가 누락 될 가능성이 있습니다.

# 로직을 어떻게 할것인가 /  CNN 


# 결측치 전처리 및 정규화 하기 


# 수치 데이터 가 아닌 것 / Id, groupId,  matchId, matchType

# 어떤 방식으로 분석을 실시 해야 할 것인가??
# 일단 솔로와 / 듀오와 / 스쿼드별로 경기를 분리해서 분석을 실시해야한다. 
# 일단 나는 솔로만 나눠서 해볼꺼야

# 솔로/ 스쿼드 / 듀오 나누기 코드


train.loc[train["matchType"] == "solo-fpp", "matchType"] = 0  # matchType 컬럼에 있는 solo값을 0으로 바꿔
train.loc[train["matchType"] == "solo", "matchType"] = 0  # matchType 컬럼에 있는 solo값을 0으로 바꿔
train.loc[train["matchType"] == "duo-fpp", "matchType"] = 1 # matchType 컬럼에 있는 duo값을 0으로 바꿔
train.loc[train["matchType"] == "duo", "matchType"] = 1 # matchType 컬럼에 있는 duo값을 0으로 바꿔
train.loc[train["matchType"] == "squad-fpp", "matchType"] = 2 # matchType 컬럼에 있는 squad 값을 0으로 바꿔
train.loc[train["matchType"] == 'squad', "matchType"] = 2 # matchType 컬럼에 있는 squad 값을 0으로 바꿔
train.loc[train["matchType"] == 'normal-squad-fpp', "matchType"] = 2 # matchType 컬럼에 있는 squad 값을 0으로 바꿔

temp = pd.DataFrame(train.groupby("matchId").size(), columns=["player"]) # matchId컬럼을 기준으로 개수를 나타내고 그 컬럼이름을 player로 하고 / 이를 새롭게 데이터프레임으로  만들고 empp값으롷 넣어
print('템프', temp)


# print("Type: ", pd.unique(train.matchType), "\nCount: ", len(pd.unique(train.matchType)))


# print(train.describe())
train = train.merge(temp, left_on="matchId", right_on="matchId") # 테이블 병합 / 
# print(train)

solo = train.loc[train.matchType == 0]
duo = train.loc[train.matchType == 1]
squad = train.loc[train.matchType == 2]
     
print(solo)

print(solo.corr())

# 이제 해야 할 것 전처리 / 정규화 / 결측치 없음 / 정규화는 어떻게 할 것인가 

# 만약에 전처리와 정규화를 다 완료했다는 가정하에 그렇다면 나는 어떤 방식으로 분석을 해야 할 것 인가

# 상관분석, 회기 분석, CNN, 의사결정 트리,  1,2,3,4 / 
