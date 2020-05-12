function pushDemo(GridWorldTask){
    let task = new GridWorldTask({
        container: $("#taskinfo4")[0],
        reward_container: $("#rewardinfo4")[0],
        step_callback: (d) => {console.log(d)},
        endtask_callback: (result_data,r) => {
            pushDemo(GridWorldTask);
        }
    });
    task.init({
        init_state: {
            'agent': [2,2],
            'cargo1': [3,2],
            'cargo2': [1,1],
            'train': [6,6],
            'trainvel': [1,0]
        },
        switch_pos: [7,7],
        targets: {
            'target1': [3,3],
            'target2': [1,0]
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: 3,
        reset: true,
    });
    task.start();
}