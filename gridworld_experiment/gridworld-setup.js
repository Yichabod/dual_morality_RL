function clickStart(hide, show){
    document.getElementById(hide).style.display="none";
    document.getElementById(show).style.display = "block";
    window.scrollTo(0,0);
}


const num_training = 60
const num_test = 30
const num_total = 90

function test_info(){
    clickStart('page1','page2')
    var test_info = "You have completed " + String(num_training) + "/" + String(num_total) + " trials. ";
    test_info += "For the last " + String(num_test) + " trials, you will be placed under a time constraint.\r\n\r\n";   
    test_group = Math.floor(Math.random() * 2);
    if (test_group==0){
        test_info += "You should take 10 seconds to look at the board and plan your moves. When 10 seconds is up, the counter will turn green and you can then take your 5 steps" 
    } else {
        test_info += "You will have a time limit of 5 seconds to complete the board. There will be a counter on the right displaying your time remaining. If you do not complete all 5 moves before time runs out, you will get the lowest possible reward of -4"
    }
    document.
    getElementById('test_info').innerText = test_info  
}

function run_train(data,GridWorldTask,num=60,idxs=undefined) {
    if (idxs == undefined){
        idxs = Array.apply(null, {length: num_training}).map(Number.call, Number)
    }
    
    idx = Math.floor(Math.random() * idxs.length);
    console.log(idx)
    idxs.splice(idxs.indexOf(idx), 1);
    
    document.getElementById('tasknum').innerText = "Trial " + String(num) + "/" + String(num_total);
    trial_data = data[idx]
    let task = new GridWorldTask({
        container: $("#task")[0],
        step_callback: (d) => {},
        endtask_callback: (trial_data,r) => {
            saveData(num, trial_data, r, "train")
            if (num >= num_training){
                test_info();
            }
            else {run_train(data,GridWorldTask,num+1,idxs)}
        }
    });
    task.init({
        init_state: {
            'agent': trial_data['agent'],
            'cargo1': trial_data['cargo1'],
            'cargo2': trial_data['cargo2'],
            'train': trial_data['train'],
            'trainvel': trial_data['trainvel']
        },
        switch_pos: trial_data['switch'],
        targets: {
            'target1': trial_data['target1'],
            'target2': trial_data['target2']
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: trial_data['best_reward']
    });
    task.start();
}

function run_test(data,GridWorldTask,test_group,num=1,idxs=undefined) {
    document.getElementById('tasknum').innerText = "Trial " + String(num_training+num) + "/" + String(num_total);

    var wait_time
    var time_limit
    
    test_group = 1
    if (test_group == 0){ //time delay group
        wait_time = 10;
    } else { //time pressure group
        wait_time = 0;
        time_limit = 5;
    }

    if (idxs == undefined){
        idxs = Array.apply(null, {length: num_test}).map(Number.call, Number)
    }
    idx = Math.floor(Math.random() * idxs.length);
    idx = idxs[idx]
    idxs.splice(idxs.indexOf(idx), 1);
    trial_data = data[idx]
    console.log(idxs)
    
    task = new GridWorldTask({
        container: $("#task")[0],
        step_callback: (d) => {},
        endtask_callback: (trial_data,r) => {
            saveData(num, trial_data, r, "test", test_group)
            if (num >= num_test){clickStart('page1','page3')}
            else {run_test(data,GridWorldTask, test_group, num+1, idxs)}
        }
    });
    task.init({
        init_state: {
            'agent': trial_data['agent'],
            'cargo1': trial_data['cargo1'],
            'cargo2': trial_data['cargo2'],
            'train': trial_data['train'],
            'trainvel': trial_data['trainvel']
        },
        switch_pos: trial_data['switch'],
        targets: {
            'target1': trial_data['target1'],
            'target2': trial_data['target2']
        },
        show_rewards: true,
        wait_time: wait_time,
        time_limit: time_limit,
        best_reward: trial_data['best_reward']
    });
    task.start();
}

function saveData(num, trial_data, r, type, time_condition = undefined) {
    var datajson = {};

    for (i = 0; i < 5; i++){
        var data = trial_data[i];
        var step = i+1;
        var action = undefined;
        var millis = undefined;
        var reward_step = undefined;
        var reward_cum = undefined;
        var hitswitch = undefined;
        var push1 = undefined;
        var push2 = undefined;
        var hitagent = undefined;
        var hit1 = undefined;
        var hit2 = undefined;
        var get1 = undefined;
        var get2 = undefined
        var state = undefined;

        if (data != undefined){
            action = data[0];
            millis = data[1];
            reward_step = data[2]
            reward_cum = data[3]
            hitswitch = data[4]
            push1 = data[5]
            push2 = data[6]
            hitagent = data[7]
            hit1 = data[8]
            hit2 = data[9]
            get1 = data[10]
            get2 = data[11]
            state = data[12]
        }
        datajson[i] = {
            'userid': 1,
            'trial': num,
            'type': type,
            'timed': time_condition,
            'step': step,
            'action': action,
            'reaction_millis': millis,
            'reward_step': reward_step,
            'reward_cum': reward_cum,
            'hitswitch': hitswitch,
            'push1': push1,
            'push2': push2,
            'hitagent': hitagent,
            'hit1': hit1,
            'hit2': hit2,
            'get1': get1,
            'get2': get2,
            'state': state
        }
    }

    datajson[5] = {
        'userid': 1,
        'trial': num,
        'type': type,
        'timed': time_condition,
        'step': 6,
        'action': undefined,
        'reaction_millis': undefined,
        'reward_step': undefined,
        'reward_cum': r,
        'push1': undefined,
        'push2': undefined,
        'hitagent': undefined,
        'hit1': undefined,
        'hit2': undefined,
        'get1': undefined,
        'get2': undefined,
        'state': undefined
    }

    datajson = JSON.stringify(datajson)

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'write_data.php'); // change 'write_data.php' to point to php script.
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
      if(xhr.status == 200){
        console.log(xhr.responseText);
        //var response = JSON.parse(xhr.responseText);
        //console.log(response.success);
      }
    };
    xhr.send(datajson);
  }