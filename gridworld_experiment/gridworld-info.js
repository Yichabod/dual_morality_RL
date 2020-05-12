info4_done = false;

function pushDemo(GridWorldTask){
    document.getElementById('next45').disabled = true
    let task = new GridWorldTask({
        reset: true,
        container: $("#taskinfo4")[0],
        reward_container: $("#rewardinfo4")[0],
        step_callback: (d) => {console.log(d)},
        endtask_callback: (result_data,r) => {
            pushDemo(GridWorldTask);
            if (r == 3){
                document.getElementById("next45").disabled = false
            }
        }
    });
    task.init({
        init_state: {
            'agent': [1,2],
            'cargo1': [2,2],
            'cargo2': [1,1],
            'train': [6,6],
            'trainvel': [1,0]
        },
        switch_pos: [7,7],
        targets: {
            'target1': [2,3],
            'target2': [1,0]
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: 3,
    });
    task.start();
}