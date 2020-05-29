function clickStart(hide, show){
    if (hide!="consent" || document.getElementById("consent_checkbox").checked){
        document.getElementById(hide).style.display="none";
        document.getElementById(show).style.display = "block";
        window.scrollTo(0,0);
    } 
    if (show == 'finishpage'){
        var bonus = total_score * 0.05
        bonus = Math.round(100*bonus)/100
        if (bonus < 0){ bonus = 0 }
        document.getElementById("completioncode").innerHTML =  "Secret Completion Code: A8M9KF22PXKS"
        document.getElementById("bonusmsg").innerHTML = "You will recieve a bonus based on your total score for the task. Your total was " + parseInt(total_score) + ", providing a bonus of $" + parseFloat(bonus) + ". Please enter your mTurk ID so that we can correctly assign your bonus. Please be aware that this may take some time to process."    
    }
}

const num_training = 60
const num_test = 30
const num_total = 90


function test_info(GridWorldTask){
    var test_info = "You have completed " + String(num_training) + "/" + String(num_total) + " trials. ";
    test_info += "For the last " + String(num_test) + " trials, you will be placed under a time constraint.\r\n\r\n";   
    test_group = Math.floor(Math.random() * 2);
    if (test_group==0){
        test_info += "You should take 7 seconds to look at the board and plan your moves. When 7 seconds is up, the counter will turn green and you can then take your 5 steps<br><br>Try one practice trial below, and get the best score to continue!" 
    } else {
        test_info += "You will have a time limit of 7 seconds to complete the board. There will be a counter on the right displaying your time remaining. <b>If you do not complete all 5 moves before time runs out, you will get the lowest possible reward of -4</b><br><br>Try one practice trial below, and get the best score to continue!"
    }
    clickStart('page1','testinfo')
    document.getElementById('test_info').innerHTML = test_info  
}

var total_score = 0;

function run_train(data,GridWorldTask,num=1,idxs=undefined) {
    if (idxs == undefined){
        idxs = Array.apply(null, {length: num_training}).map(Number.call, Number)
    }
    
    idx = Math.floor(Math.random() * idxs.length);
    idx = idxs[idx]
    console.log(idx)
    idxs.splice(idxs.indexOf(idx), 1);

    document.getElementById('tasknum').innerText = "Trial " + String(num) + "/" + String(num_total);
    document.getElementById('totalscore').innerText = "Total Score: " + String(total_score);
    trial_data = data[idx]
    let task = new GridWorldTask({
        reset: false,
        container: $("#task")[0],
        reward_container: $("#reward")[0],
        step_callback: (d) => {},
        endtask_callback: (result_data,r) => {
            total_score += r;
            saveData(num, idx, result_data, r, trial_data['best_reward'],"train")
            if (num >= num_training){
                document.getElementById('test_button').style.color = "red";
                document.getElementById('test_button').disabled = true;
                test_info(GridWorldTask);
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
    document.getElementById('totalscore').innerText = "Total Score: " + String(total_score);

    var wait_time
    var time_limit

    if (test_group == 0){ //time delay group
        wait_time = 7;
    } else { //time pressure group
        wait_time = 0;
        time_limit = 7;
    }

    if (idxs == undefined){
        idxs = Array.apply(null, {length: num_test}).map(Number.call, Number)
    }
    idx = Math.floor(Math.random() * idxs.length);
    idx = idxs[idx]
    console.log(idx)
    idxs.splice(idxs.indexOf(idx), 1);
    trial_data = data[idx]
    console.log(idxs)
    
    task = new GridWorldTask({
        reset: false,
        container: $("#task")[0],
        reward_container: $("#reward")[0],
        time_container: $("#timer")[0],
        step_callback: (d) => {},
        endtask_callback: (result_data,r) => {
            total_score += r;
            saveData(num+num_training, idx, result_data, r, trial_data['best_reward'],"test", test_group)
            if (num >= num_test){clickStart('page1','finishpage')}
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

var userid = -1;

function saveData(num, idx, trial_data, r, rmax, type, time_condition = undefined) {
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
            'userid': userid,
            'trialnum': num,
            'gridnum': idx,
            'type': type,
            'timed': time_condition,
            'step': step,
            'action': action,
            'reaction_millis': millis,
            'reward_step': reward_step,
            'reward_cum': reward_cum,
            'reward_max': undefined,
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
        'userid': userid,
        'trialnum': num,
        'gridnum': idx,
        'type': type,
        'timed': time_condition,
        'step': 6,
        'action': undefined,
        'reaction_millis': undefined,
        'reward_step': undefined,
        'reward_cum': r,
        'reward_max': rmax,
        'hitswitch': undefined,
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
        var response = JSON.parse(xhr.responseText); 
        userid = response["userid"];
        console.log(userid);
      }
    };
    xhr.send(datajson);
  }