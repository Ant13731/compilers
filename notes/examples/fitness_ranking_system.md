```python

# Carrier sets
people: set[Person]
gyms: set[Gym]
machines: set[Machine]
raw_records: set[FitnessMachineRecord]
daily_records: set[FitnessMachineDaily]
daily_rankable_records: set[BlendedFitnessRankDaily]
weekly_rankable_records: set[BlendedFitnessRankWeekly]
ranking_enum: enum = {time_spent, total_reps, total_weight, blended_effort, blended_skill}

# Relationships
members: Person -/-> Gym # person is a member of at most one gym
trainers: Person <--> Person # one trainer can be assigned to multiple people, but each person should only have one trainer
    with one_to_many
employees: Person -/-> Gym # person is an employee at a Gym
managers: Person -> Person # person manages employees
machines_available: Gym <--> Machine # gyms can have many machines, machines are not unique so they can exist in multiple gyms
baseline_reps_per_minute: Machine --> int # lowest number of reps needed to qualify for leaderboards on a specific machine

# Record types
# Basic record as a measure of effort
record FitnessMachineRecord:
    # Record identifiers
    member: Person # person who did the exercise
    machine: Machine # exercise machine type
    gym: Gym # gym the exercise was done at
    start_time: Timestamp # time we started using the machine
    end_time: Timestamp # time we ended using the machine
    # Effort data
    reps: int # reps per set * number of sets
    weight: int # weight in lbs
    # Metadata
    with_trainer: Person | None # the trainer, if the person had a trainer while using the machine

record FitnessMachineDaily:
    # Record identifiers
    member: Person # person who did the exercise
    machine: Machine # exercise machine type
    gym: Gym # gym the exercise was done at
    day: Timestamp
    # Effort data
    time_spent: int # seconds
    total_reps: int
    total_weight: int
    # then we judge effort based on total_reps * total_weight * time_spent
    # then we judge skill based on total_reps * total_weight / time_spent

record BlendedFitnessRankDaily:
    # Record identifiers
    member: Person # person who did the exercise
    gym: Gym # gym the exercise was done at
    day: Timestamp
    # Effort data
    time_spent: int # seconds
    total_reps: int
    total_weight: int
    blended_effort: float # combine records across machines into one value per person
    blended_skill: float

record BlendedFitnessRankWeekly:
    # Record identifiers
    member: Person # person who did the exercise
    gym: Gym # gym the exercise was done at
    week_start: Timestamp
    # Effort data
    time_spent: int # seconds
    total_reps: int
    total_weight: int
    blended_effort: float # combine records across machines into one value per person
    blended_skill: float


# Queries
procedure get_leaderboard(records: set[BlendedFitnessRankWeekly], order_by: RankingEnum) -> sequence[Person]:
    # from largest to lowest
    sorted_records: sequence[BlendedFitnessRankWeekly] = sort(records, r -> -get_value(r, order_by))
    return [r . r in sorted_records | r.member]

procedure get_leaderboard_amount_improved(this_week: set[BlendedFitnessRankWeekly], last_week: set[BlendedFitnessRankWeekly], order_by: RankingEnum) -> sequence[Person]:
    last_week_by_key: BlendedWeeklyKey -/-> BlendedFitnessRankWeekly = {r . r in records | weekly_key(r) -> r}
    sorted_records: sequence[BlendedFitnessRankWeekly] =
        sort(
            this_week,
            r -> -(get_value(last_week_by_key(last_week_by_key(r)), order_by) - get_value(r, order_by))
        )
    return [r . r in sorted_records | r.member]

# Main data processing functions
procedure raw_to_daily(records: set[FitnessMachineRecord]) -> set[FitnessMachineDaily]:
    # Assume records never cross over day boundaries for the sake of consistency (instead those are already split into two records)
    records_by_merged_key: DailyKey <-> FitnessMachineRecord = {r . r in records | extract_daily_key_from_raw(r) -> r}
    records_grouped_by_merged_key: DailyKey -> set[FitnessMachineRecord] = group_by_fst(records_by_merged_key)
    merged_records: set[FitnessMachineDaily] = {d,rs . d->rs in records_grouped_by_day | merge_raw_to_daily(d, rs)}
    return merged_records

procedure daily_to_blended_daily(records: set[FitnessMachineDaily]) -> set[BlendedFitnessRankDaily]:
    records_by_merged_key: BlendedDailyKey <-> BlendedFitnessRankDaily = {r . r in records | extract_blended_key_from_daily(r) -> r}
    records_grouped_by_merged_key: BlendedDailyKey -> set[BlendedFitnessRankDaily] = group_by_fst(records_by_merged_key)
    merged_records: set[BlendedFitnessRankDaily] = {d,rs . d->rs in records_grouped_by_day | blend_daily(d, rs)}
    return merged_records

procedure blended_daily_to_blended_weekly(records: set[BlendedFitnessRankDaily]) -> set[BlendedFitnessRankWeekly]:
    records_by_merged_key: BlendedWeeklyKey <-> BlendedFitnessRankWeekly = {r . r in records | extract_weekly_key_from_daily(r) -> r}
    records_grouped_by_merged_key: BlendedWeeklyKey -> set[BlendedFitnessRankWeekly] = group_by_fst(records_by_merged_key)
    merged_records: set[BlendedFitnessRankWeekly] = {d,rs . d->rs in records_grouped_by_day | blend_weekly(d, rs)}
    return merged_records

# Helpers/formatters
procedure extract_daily_key_from_raw(record: FitnessMachineRecord) -> DailyKey:
    return (
        record.member,
        record.machine,
        record.gym,
        start_of_day(record.start_time)
    )
procedure extract_blended_key_from_daily(record: FitnessMachineDaily) -> BlendedDailyKey:
    return (
        record.member,
        record.gym,
        record.day
    )
procedure extract_weekly_key_from_daily(record: BlendedFitnessRankDaily) -> BlendedWeeklyKey:
    return (
        record.member,
        record.gym,
        start_of_week(record.day)
    )
procedure group_by_fst(records: K <-> V) -> (K -> sequence[V]):
    res = defaultdict(list)
    for k, v in records:
        res[k].append(v)
    return res
procedure merge_raw_to_daily(daily_key: DailyKey, records: set[FitnessMachineRecord]) -> FitnessMachineDaily:
    return FitnessMachineDaily(
        daily_key.member
        daily_key.machine
        daily_key.gym
        daily_key.day
        sum({r . r in records | r.end_time - r.start_time})
        sum({r . r in records | r.reps})
        sum({r . r in records | r.weight})
    )
procedure blend_daily(blended_key: BlendedDailyKey, records: set[FitnessMachineDaily]) -> BlendedFitnessRankDaily:
    return BlendedFitnessRankDaily(
        blended_key.member,
        blended_key.gym,
        blended_key.day,
        sum({r . r in records | r.time_spent})
        sum({r . r in records | r.total_reps})
        sum({r . r in records | r.total_weight})
        sum({r . r in records | r.total_reps * r.total_weight * r.time_spent})
        sum({r . r in records | r.total_reps * r.total_weight / r.time_spent})
    )
procedure blend_daily(blended_key: BlendedWeeklyKey, records: set[BlendedFitnessRankDaily]) -> BlendedFitnessRankWeekly:
    return BlendedFitnessRankWeekly(
        blended_key.member,
        blended_key.gym,
        blended_key.week_start,
        sum({r . r in records | r.time_spent})
        sum({r . r in records | r.total_reps})
        sum({r . r in records | r.total_weight})
        sum({r . r in records | r.blended_effort})
        sum({r . r in records | r.blended_skill})
    )

```

Queries:
- Leaderboard questions in the form of 1-person queries:
    - What is my score/rank for today/this week/this year/all time?
        - by weight
        - by time spent
        - by reps completed
        - by reps per minute
        - by blended rep-weights
        - by amount improved since last week
    - How much time have I spent at the gym/machine today/week/year/all time?
    - How many reps have I completed ...?
    - How much weight have I lifted ...?
    -