info4_done = false;
info5_done = false;
info6_done = false;

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
    if (!info5_done){
    document.getElementById('next56').style.color = "red";
    document.getElementById('next56').disabled = true;
    }
        let task = new GridWorldTask({
        reset: true,
        container: $("#taskinfo5")[0],
        reward_container: $("#rewardinfo5")[0],
        step_callback: (d) => {
            if (d['reward'] == -1 && d['iter']==5){
                document.getElementById("next56").disabled = false
                info5_done = true;
                document.getElementById('next56').style.color = "white"
            }
        },
        endtask_callback: (result_data,r) => {
            trainDemo(GridWorldTask);
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
            'target1': [2,3],
            'target2': [1,0]
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: -1,
    });
    task.start();
}
function switchDemo(GridWorldTask){
    if (!info6_done){
    document.getElementById('next67').style.color = "red";
    document.getElementById('next67').disabled = true;
    }
        let task = new GridWorldTask({
        reset: true,
        container: $("#taskinfo6")[0],
        reward_container: $("#rewardinfo6")[0],
        step_callback: (d) => {
            if (d['reward'] == 0 && d['iter']==5){
                document.getElementById("next67").disabled = false
                info6_done = true;
                document.getElementById('next67').style.color = "white"
            }
        },
        endtask_callback: (result_data,r) => {
            switchDemo(GridWorldTask);
        }
            });
        task.init({
        init_state: {
            'agent': [2,3],
            'cargo1': [4,2],
            'cargo2': [6,6],
            'train': [1,2],
            'trainvel': [1,0]
        },
        switch_pos: [4,3],
        targets: {
            'target1': [2,3],
            'target2': [2,4]
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: 0,
    });
    task.start();
}


