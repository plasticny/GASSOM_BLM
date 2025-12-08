function [subjects] = getRandSubject (n, r)
    cipic_subjects = [ ...
        3,  8,  9, 10, 11, 12, 15, 17, 18, 19,  ...
        20, 21, 27, 28, 33, 40, ...
        44, 48, 50, 51, 58, 59, ...
        60, 61, 65, 119, 124, 126, 127, 131, 133, 134, ...
        135, 137, 147, 148, 152, 153, 154, 155, 156, 158, ...
        162, 163, 165 ...
    ];

    if r ~= -1
        cipic_subjects = cipic_subjects(cipic_subjects ~= r);
    end

    l = length(cipic_subjects);

    perm = randperm(l);
    perm = perm(1:n);
    subjects = sort(cipic_subjects(perm));
end

clc; clear all; addpath(genpath(pwd));

rng(50);

individual_subjects = getRandSubject(10, -1);
disp(individual_sujects);

non_individual_subjects = zeros(length(individual_subjects), 10);
for i = 1:length(individual_subjects)
    non_individual_subjects(i, :) = getRandSubject(10, individual_subjects(i));
end
disp(non_individual_subjects);

save("wenzel_cipic_subject/wenzel_cipic_subject.mat", "individual_subjects", "non_individual_subjects");
