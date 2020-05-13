info4_done = false;

function clearGrid(div){
    myNode = document.getElementById(div);
    while (myNode.firstChild) {
        myNode.removeChild(myNode.lastChild);
    }
}

function pushDemo(GridWorldTask){
    if (!info4_done){
        document.getElementById('next45').disabled = true
        document.getElementById('next45').style.color = "red"
    }
    let task = new GridWorldTask({
        reset: true,
        container: $("#taskinfo4")[0],
        reward_container: $("#rewardinfo4")[0],
        step_callback: (d) => {
            if (d['reward'] == 3 && d['iter']==5){
                document.getElementById("next45").disabled = false
                info4_done = true;
                document.getElementById('next45').style.color = "white"
            }
        },
        endtask_callback: (result_data,r) => {
            pushDemo(GridWorldTask);
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
function trainDemo(GridWorldTask){
    document.getElementById('next45').disabled = true
    let task = new GridWorldTask({
        reset: true,
        container: $("#taskinfo5")[0],
        reward_container: $("#rewardinfo5")[0],
        step_callback: (d) => {console.log(d)},
        endtask_callback: (result_data,r) => {
            pushDemo(GridWorldTask);
            if (r == 3){
                document.getElementById("next45").disabled = false
            }//button name needs change
        }
    });
    task.init({
        init_state: {
            'agent': [2,2],
            'cargo1': [4,1],
            'cargo2': [1,4],
            'train': [1,1],
            'trainvel': [1,0]
        },
        switch_pos: [7,7],
        targets: {
            'target1': [6,6],
            'target2': [7,6]
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: -1,
    });
    task.start();
}
function switchDemo(GridWorldTask){
    document.getElementById('next45').disabled = true
    let task = new GridWorldTask({
        reset: true,
        container: $("#taskinfo6")[0],
        reward_container: $("#rewardinfo6")[0],
        step_callback: (d) => {console.log(d)},
        endtask_callback: (result_data,r) => {
            pushDemo(GridWorldTask);
            if (r == 3){
                document.getElementById("next45").disabled = false
            }//button name needs change
        }
    });
    task.init({
        init_state: {
            'agent': [2,3],
            'cargo1': [4,2],
            'cargo2': [1,4],
            'train': [1,2],
            'trainvel': [1,0]
        },
        switch_pos: [4,3],
        targets: {
            'target1': [6,6],
            'target2': [7,6]
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: -1,
    });
    task.start();
}


